#!/usr/bin/env python3
"""졸음 자동 라벨링 (MediaPipe Face Landmarker + 규칙).

단계
  extract   영상 -> 프레임별 특징 JSON (EAR, MAR, 고개 각도, 눈깜빡임 점수, 얼굴 검출 여부)
  calibrate 손으로 만든 라벨(_졸음라벨초안.json)과 맞춰 보며 임계값을 찾는다
  label     찾은 임계값으로 클립마다 구간 라벨(JSON/CSV)을 만든다
  selftest  MediaPipe 없이 라벨 규칙 로직만 점검한다
  all       extract -> (calibrate) -> label 을 한 번에 (X, y 가 함께 저장된다)

상태 이름은 손 라벨과 같게 맞춘다:
  정상(눈 뜸) / 눈감김 / 고개숙임(눈 뜸) / 하품 / 불확실(얼굴 없음)
손 라벨은 눈감김과 고개숙임을 '눈감김·고개숙임' 하나로 묶었으므로, 손 라벨과 비교할 때만 둘을 합쳐 채점한다.
"""
import argparse, glob, itertools, json, math, os, sys, urllib.request
import numpy as np
import pandas as pd

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
R_EYE = [33, 160, 158, 133, 153, 144]      # 눈 6점 (기획서의 좌표 17개 중 12개)
L_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = (61, 291, 13, 14)                  # 입꼬리 2 + 안쪽 윗입술/아랫입술 (4개), 코끝은 1번
S_OK, S_CLOSE, S_HEAD, S_YAWN, S_UNC = "정상(눈 뜸)", "눈감김", "고개숙임(눈 뜸)", "하품", "불확실(얼굴 없음)"
S_MANUAL_CLOSE = "눈감김·고개숙임"          # 손 라벨에서 눈감김과 고개숙임을 묶은 상태 이름
MANUAL_SUFFIX = "_졸음라벨초안.json"
DEFAULT = dict(eye_mode="ear", ear_ratio=0.65, blink_th=0.5, pitch_dev=20.0,
               mar_th=0.5, min_close=0.25, min_yawn=0.8, lost_close=False,
               warn_s=1.5, sleep_s=3.0)   # 눈감김 연속 시간 기준: 주의(마이크로슬립)·졸음(수면), Euro NCAP SD-202 참고
STAGES = ["정상", "주의", "졸음"]          # stage 인덱스 순서
COLS = ["frame", "t_s", "face", "ear_l", "ear_r", "ear", "mar", "pitch", "yaw", "roll",
        "blink_l", "blink_r", "jaw_open"]
X_COLS = COLS[2:]                          # 학습 입력 특징 (frame, t_s 제외)
STATES = [S_OK, S_CLOSE, S_HEAD, S_YAWN, S_UNC]    # y 인덱스 순서


# ───────────────────────── 1) 특징 추출 (MediaPipe 필요) ─────────────────────────
def ensure_model(path):
    if not os.path.exists(path):
        print("얼굴 랜드마크 모델 다운로드:", MODEL_URL)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def _ear(p, idx):
    q = [p[i] for i in idx]
    d = math.dist
    return (d(q[1], q[5]) + d(q[2], q[4])) / (2 * d(q[0], q[3]) + 1e-9)


def extract_video(path, model_path, roi=None):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    opts = vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3)
    lm = vision.FaceLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if roi:
            h, w = f.shape[:2]
            x0, y0, x1, y1 = roi
            f = f[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        h, w = f.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                  int(round(i * 1000 / fps)))
        t = i / fps
        if res.face_landmarks:
            pts = [(p.x * w, p.y * h) for p in res.face_landmarks[0]]
            el, er = _ear(pts, L_EYE), _ear(pts, R_EYE)
            a, b, c, d = MOUTH
            mar = math.dist(pts[c], pts[d]) / (math.dist(pts[a], pts[b]) + 1e-9)
            pitch = yaw = roll = np.nan
            if res.facial_transformation_matrixes:
                R = np.array(res.facial_transformation_matrixes[0])[:3, :3]
                sy = math.hypot(R[0, 0], R[1, 0])
                pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
                yaw = math.degrees(math.atan2(-R[2, 0], sy))
                roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
            bs = {c.category_name: c.score for c in (res.face_blendshapes[0] if res.face_blendshapes else [])}
            rows.append([i, t, 1, el, er, (el + er) / 2, mar, pitch, yaw, roll,
                         bs.get("eyeBlinkLeft", np.nan), bs.get("eyeBlinkRight", np.nan),
                         bs.get("jawOpen", np.nan)])
        else:
            rows.append([i, t, 0] + [np.nan] * 10)
        i += 1
    cap.release()
    lm.close()
    return pd.DataFrame(rows, columns=COLS), fps


def cmd_extract(a):
    model = ensure_model(a.model)
    os.makedirs(a.out, exist_ok=True)
    vids = sorted(glob.glob(os.path.join(a.video_root, "**", "*.mp4"), recursive=True))
    if a.only_labeled_in:
        have = {os.path.basename(p)[:-len(MANUAL_SUFFIX)]
                for p in glob.glob(os.path.join(a.only_labeled_in, "**", "*" + MANUAL_SUFFIX), recursive=True)}
        vids = [v for v in vids if os.path.splitext(os.path.basename(v))[0] in have]
    roi = tuple(float(x) for x in a.roi.split(",")) if a.roi else None
    print(f"처리할 영상 {len(vids)}개")
    for k, v in enumerate(vids, 1):
        stem = os.path.splitext(os.path.basename(v))[0]
        dst = os.path.join(a.out, stem + ".json")
        if os.path.exists(dst) and not a.overwrite:
            continue
        df, fps = extract_video(v, model, roi)
        json.dump(dict(clip=stem, fps=fps, columns=COLS,
                       frames=df.astype(object).where(df.notna(), None).values.tolist()),
                  open(dst, "w", encoding="utf-8"))
        print(f"[{k}/{len(vids)}] {stem}: {len(df)}프레임, 얼굴 검출 {df.face.mean():.0%}")


# ───────────────────────── 2) 규칙 라벨링 (MediaPipe 불필요) ─────────────────────────
def runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def drop_short(mask, min_frames):
    m = mask.copy()
    for a, b in runs(mask):
        if b - a + 1 < min_frames:
            m[a:b + 1] = False
    return m


def label_frames(df, fps, p):
    """프레임별 상태 배열을 돌려준다."""
    n = len(df)
    face = df["face"].values.astype(bool)
    if p["eye_mode"] == "ear":
        ear = np.nan_to_num(df["ear"].values, nan=9.0)
        base = np.nanpercentile(df["ear"].values[face], 75) if face.sum() > 5 else 0.25
        closed = face & (ear < p["ear_ratio"] * base)          # 이 클립의 '눈 뜬' 기준 대비 비율
    else:
        bl = np.nan_to_num((df["blink_l"].values + df["blink_r"].values) / 2, nan=0.0)
        closed = face & (bl > p["blink_th"])
    head = np.zeros(n, bool)
    if p["pitch_dev"] < 900 and face.sum() > 5:
        pitch = df["pitch"].values
        med = np.nanmedian(pitch[face])
        head = face & (np.abs(np.nan_to_num(pitch, nan=med) - med) > p["pitch_dev"])
    minc = max(1, int(round(p["min_close"] * fps)))
    close_all = drop_short(closed | head, minc)
    lost = drop_short(~face, minc) if p["lost_close"] else np.zeros(n, bool)
    mar = np.nan_to_num(df["mar"].values, nan=0.0)
    yawn = drop_short(face & (mar > p["mar_th"]), max(1, int(round(p["min_yawn"] * fps))))
    state = np.full(n, S_OK, dtype=object)
    state[~face] = S_UNC
    state[lost] = S_HEAD                       # 얼굴이 사라진 구간은 고개를 크게 숙인 것으로 본다
    state[close_all & ~closed] = S_HEAD        # 눈은 떴는데 고개만 벗어남
    state[close_all & closed] = S_CLOSE
    state[yawn] = S_YAWN
    return state


def eye_stage(state, fps, p):
    """눈감김이 연속 warn_s 에 닿은 프레임부터 주의(1), sleep_s 에 닿은 프레임부터 졸음(2).
    고개숙임은 보조 신호라 시간에 넣지 않는다."""
    stage = np.zeros(len(state), dtype=int)
    for a, b in runs(state == S_CLOSE):
        el = (np.arange(b - a + 1) + 1) / fps          # 눈 감기 시작부터 지난 시간
        stage[a:b + 1] = np.where(el >= p["sleep_s"], 2, np.where(el >= p["warn_s"], 1, 0))
    return stage


def to_segments(state, fps):
    segs, i = [], 0
    for s, g in itertools.groupby(enumerate(state), key=lambda x: x[1]):
        g = list(g)
        segs.append({"start_s": round(g[0][0] / fps, 3), "end_s": round((g[-1][0] + 1) / fps, 3), "state": s})
    return segs


def load_features(path):
    d = json.load(open(path, encoding="utf-8"))
    return pd.DataFrame(d["frames"], columns=d["columns"]).astype(float), d["fps"]


# ───────────────────────── 3) 손 라벨과 맞춰 보정 ─────────────────────────
def load_manual(root):
    out = {}
    for pth in glob.glob(os.path.join(root, "**", "*" + MANUAL_SUFFIX), recursive=True):
        d = json.load(open(pth, encoding="utf-8"))
        out[os.path.basename(pth)[:-len(MANUAL_SUFFIX)]] = d["timeline"]
    return out


def auto_at(state, fps, t, half=0.25):
    a, b = int(round((t - half) * fps)), int(round((t + half) * fps))
    w = [s for s in state[max(0, a):b + 1]]
    if not w:
        return S_UNC
    vals, cnt = np.unique(np.array(w, dtype=object), return_counts=True)
    return vals[cnt.argmax()]


def score(feats, manual, p):
    """손 라벨 시점마다 자동 상태를 비교한다. 정상·눈감김·하품만 채점(손 동작·불확실은 제외)."""
    classes = [S_OK, S_MANUAL_CLOSE, S_YAWN]
    tp = {c: 0 for c in classes}; fp = {c: 0 for c in classes}; fn = {c: 0 for c in classes}
    lost_close = [0, 0]
    for stem, (df, fps) in feats.items():
        if stem not in manual:
            continue
        st = label_frames(df, fps, p)
        st = np.where((st == S_CLOSE) | (st == S_HEAD), S_MANUAL_CLOSE, st)   # 손 라벨처럼 합쳐서 비교
        for r in manual[stem]:
            if r["state"] not in classes:
                continue
            pred = auto_at(st, fps, r["t_s"])
            if r["state"] == S_MANUAL_CLOSE:
                lost_close[1] += 1
                lost_close[0] += int(pred == S_UNC)
            if pred == r["state"]:
                tp[pred] += 1
            else:
                fn[r["state"]] += 1
                if pred in classes:
                    fp[pred] += 1
    f1, res = [], {}
    for c in classes:
        pr = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rc = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        f1.append(f)
        res[c] = dict(precision=round(pr, 3), recall=round(rc, 3), f1=round(f, 3), n=tp[c] + fn[c])
    total = sum(tp.values()) + sum(fn.values())
    return dict(macro_f1=round(float(np.mean(f1)), 4), accuracy=round(sum(tp.values()) / max(1, total), 4),
                per_class=res, close_frames_face_lost=f"{lost_close[0]}/{lost_close[1]}")


def cmd_calibrate(a):
    manual = load_manual(a.manual_root)
    feats = {}
    for pth in glob.glob(os.path.join(a.features, "*.json")):
        stem = os.path.splitext(os.path.basename(pth))[0]
        if stem in manual:
            feats[stem] = load_features(pth)
    print(f"손 라벨과 짝이 맞는 클립 {len(feats)}개 (손 라벨 {len(manual)}개)")
    if not feats:
        sys.exit("짝이 맞는 클립이 없습니다. 파일 이름(영상 이름)과 손 라벨 이름을 확인하세요.")
    allf = pd.concat([d for d, _ in feats.values()])
    print(f"전체 얼굴 검출률 {allf.face.mean():.1%}")
    grid = []
    for mode, thr in [("ear", r) for r in (0.5, 0.6, 0.7, 0.8)] + [("blink", b) for b in (0.35, 0.5, 0.65)]:
        for pd_ in (15.0, 25.0, 999.0):
            for mar in (0.35, 0.5, 0.65):
                for mc in (0.1, 0.25, 0.4):
                    for lc in (False, True):
                        p = dict(DEFAULT, eye_mode=mode, pitch_dev=pd_, mar_th=mar, min_close=mc, lost_close=lc)
                        p["ear_ratio" if mode == "ear" else "blink_th"] = thr
                        grid.append(p)
    best, best_s = None, None
    for p in grid:
        s = score(feats, manual, p)
        if best_s is None or s["macro_f1"] > best_s["macro_f1"]:
            best, best_s = p, s
    base = score(feats, manual, DEFAULT)
    print("\n기본 설정 :", json.dumps(base, ensure_ascii=False))
    print("최적 설정 :", json.dumps(best, ensure_ascii=False))
    print("최적 결과 :", json.dumps(best_s, ensure_ascii=False))
    json.dump(dict(params=best, score=best_s, default_score=base, clips=len(feats)),
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("저장:", a.out)
    print("\n※ 같은 클립으로 고르고 같은 클립으로 점수를 냈으므로 낙관적인 값입니다. "
          "손 라벨 자체에도 ±0.5초 오차가 있습니다.")


# ───────────────────────── 4) 클립별 자동 라벨 출력 ─────────────────────────
def cmd_label(a):
    p = dict(DEFAULT)
    if a.params:
        p.update(json.load(open(a.params, encoding="utf-8"))["params"])
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for pth in sorted(glob.glob(os.path.join(a.features, "*.json"))):
        stem = os.path.splitext(os.path.basename(pth))[0]
        df, fps = load_features(pth)
        st = label_frames(df, fps, p)
        segs = to_segments(st, fps)
        stage = eye_stage(st, fps, p)
        dur = round(len(df) / fps, 3)
        json.dump(dict(clip=stem, video_duration_s=dur, fps=round(fps, 2),
                       label_source="자동(MediaPipe Face Landmarker + 규칙), 사람 검수 전", params=p,
                       face_detect_rate=round(float(df.face.mean()), 3), segments=segs,
                       classes=STATES, feature_names=X_COLS,
                       X=np.round(df[X_COLS].fillna(0.0).values, 4).tolist(),  # [프레임 수, 특징 수], 얼굴 없음은 0
                       y=[STATES.index(s) for s in st],
                       stage_classes=STAGES, stage=stage.tolist()),
                  open(os.path.join(a.out, stem + "_자동라벨.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        close_s = sum(s["end_s"] - s["start_s"] for s in segs if s["state"] == S_CLOSE)
        n_close = sum(1 for s in segs if s["state"] == S_CLOSE)
        n_head = sum(1 for s in segs if s["state"] == S_HEAD)
        n_yawn = sum(1 for s in segs if s["state"] == S_YAWN)
        longest = max([s["end_s"] - s["start_s"] for s in segs if s["state"] == S_CLOSE], default=0)
        rows.append(dict(clip=stem, 길이_s=dur, 얼굴검출률=round(float(df.face.mean()), 3),
                         눈감김_횟수=n_close, 눈감김_합계_s=round(close_s, 2), 눈감김_최장_s=round(longest, 2),
                         고개숙임_횟수=n_head, 최고단계=STAGES[stage.max(initial=0)],
                         하품_횟수=n_yawn))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "_요약.csv"), index=False, encoding="utf-8-sig")
    print(f"{len(rows)}개 클립 라벨 저장 -> {a.out} (요약: _요약.csv)")


# ───────────────────────── 5) 로직 점검 ─────────────────────────
def cmd_selftest(_):
    fps, n = 30.0, 300
    t = np.arange(n) / fps
    ear = np.full(n, 0.30); mar = np.full(n, 0.05); pitch = np.zeros(n); face = np.ones(n)
    ear[90:120] = 0.08            # 3.0~4.0초 눈감김 (1.0초)
    ear[45:48] = 0.08             # 1.5초 깜빡임 0.1초 (무시되어야 함)
    mar[180:230] = 0.8            # 6.0~7.7초 하품
    pitch[150:160] = 40           # 5.0~5.3초 고개 숙임
    face[250:260] = 0             # 얼굴 없음 0.33초
    df = pd.DataFrame(dict(frame=np.arange(n), t_s=t, face=face, ear_l=ear, ear_r=ear, ear=ear, mar=mar,
                           pitch=pitch, yaw=0.0, roll=0.0, blink_l=np.where(ear < 0.15, 0.9, 0.1),
                           blink_r=np.where(ear < 0.15, 0.9, 0.1), jaw_open=0.0))
    df.loc[face == 0, ["ear", "ear_l", "ear_r", "mar", "pitch"]] = np.nan
    for mode in ("ear", "blink"):
        st = label_frames(df, fps, dict(DEFAULT, eye_mode=mode))
        segs = to_segments(st, fps)
        got = [(s["state"], s["start_s"], s["end_s"]) for s in segs]
        assert (S_CLOSE, 3.0, 4.0) in got, (mode, got)
        assert not any(s[0] == S_CLOSE and s[1] < 2 for s in got), "깜빡임을 눈감김으로 잘못 셈"
        assert any(s[0] == S_YAWN and abs(s[1] - 6.0) < 0.1 for s in got), got
        assert any(s[0] == S_HEAD and abs(s[1] - 5.0) < 0.1 for s in got), "고개 숙임을 놓침"
        assert any(s[0] == S_UNC for s in got), "얼굴 없음 구간 표기 누락"
    # 보정 로직: 정답 타임라인이 같으면 점수가 높아야 한다
    manual = [dict(t_s=round(i * 0.5, 1),
                   state=(S_MANUAL_CLOSE if 3.0 <= i * 0.5 < 4.0 or 5.0 <= i * 0.5 < 5.3 else
                          S_YAWN if 6.2 <= i * 0.5 < 7.5 else S_OK)) for i in range(20)]
    s = score({"x": (df, fps)}, {"x": manual}, dict(DEFAULT))
    assert s["macro_f1"] > 0.8, s
    # 단계: 눈감김 연속 1.5초에 닿은 프레임부터 주의, 3초부터 졸음
    stage = eye_stage(np.array([S_CLOSE] * 100 + [S_OK] * 10, dtype=object), fps, DEFAULT)
    assert stage[43] == 0 and stage[44] == 1 and stage[88] == 1 and stage[89] == 2 and stage[100] == 0, stage
    print("selftest 통과", s["macro_f1"])


def cmd_all(a):
    """extract -> (손 라벨이 있으면 calibrate) -> label 을 한 번에 실행해 X, y 를 함께 만든다."""
    cmd_extract(a)
    params = None
    if a.manual_root:
        params = "best_params.json"
        cmd_calibrate(argparse.Namespace(features=a.out, manual_root=a.manual_root, out=params))
    cmd_label(argparse.Namespace(features=a.out, params=params, out="labels_auto"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract"); e.set_defaults(f=cmd_extract)
    r = sp.add_parser("all"); r.set_defaults(f=cmd_all)
    r.add_argument("--manual-root", help="TL1 폴더: 주면 손 라벨로 임계값을 맞춘 뒤 라벨링, 없으면 기본 임계값")
    for x in (e, r):
        x.add_argument("--video-root", required=True, help="TS1 폴더")
        x.add_argument("--out", default="features")
        x.add_argument("--model", default="face_landmarker.task")
        x.add_argument("--only-labeled-in", help="TL1 폴더: 손 라벨(_졸음라벨초안.json)이 있는 클립만 처리")
        x.add_argument("--roi", help="x0,y0,x1,y1 (0~1). 얼굴이 작으면 얼굴 쪽만 잘라서 검출, 예: 0.3,0,0.95,0.7")
        x.add_argument("--overwrite", action="store_true")
    c = sp.add_parser("calibrate"); c.set_defaults(f=cmd_calibrate)
    c.add_argument("--features", default="features"); c.add_argument("--manual-root", required=True)
    c.add_argument("--out", default="best_params.json")
    l = sp.add_parser("label"); l.set_defaults(f=cmd_label)
    l.add_argument("--features", default="features"); l.add_argument("--params")
    l.add_argument("--out", default="labels_auto")
    s = sp.add_parser("selftest"); s.set_defaults(f=cmd_selftest)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
