# 졸음 자동 라벨링

MediaPipe 얼굴 랜드마크(눈 12점 · 입 4점 · 코끝 1점)로 프레임마다 EAR·MAR·고개 각도를 계산하고, 규칙으로 프레임별 라벨을 만듭니다.
결과는 시계열 모델(1D CNN, LSTM)에 바로 넣을 수 있는 `X`(특징), `y`(상태) JSON입니다.

## 설치 (한 번만)
```
pip install -r requirements.txt
```
팀원 모두 같은 버전을 씁니다. 특히 `mediapipe==1.2.0`이 다르면 특징값이 달라질 수 있습니다.
첫 실행 때 얼굴 랜드마크 모델(`face_landmarker.task`)을 자동으로 받습니다. 인터넷이 막혀 있으면 모델 파일을 이 폴더에 직접 넣으세요.

## 실행 (이 폴더에서)
```
# 0) 로직 점검 (MediaPipe 필요 없음)
python auto_label.py selftest

# 팀원: 공유된 best_params.json으로 X, y 만들기 (권장)
python auto_label.py extract --video-root <영상 폴더>
python auto_label.py label --params best_params.json

# 임계값 담당자: 손 라벨로 임계값을 다시 맞추고 X, y까지 한 번에
python auto_label.py all --video-root <영상 폴더> --manual-root <손 라벨 폴더>
```

| 명령 | 하는 일 | 저장 |
| --- | --- | --- |
| `extract` | 영상 → 프레임별 특징 | `features/<클립>.json` |
| `calibrate` | 손 라벨(`<클립>_졸음라벨초안.json`)과 비교해 임계값 찾기 | `best_params.json` |
| `label` | 임계값으로 프레임별 라벨 생성 | `labels_auto/<클립>_자동라벨.json`, `_요약.csv` |
| `all` | `extract` → (`--manual-root`가 있으면 `calibrate`) → `label` | 위 전부 |

`extract`는 결과 파일이 이미 있으면 건너뜁니다. 다시 만들려면 `--overwrite`를 붙이세요.

## 학습 데이터 형식 (`labels_auto/<클립>_자동라벨.json`)
- `X`: `[프레임 수, 11]`. 순서는 `feature_names` (`face, ear_l, ear_r, ear, mar, pitch, yaw, roll, blink_l, blink_r, jaw_open`). 얼굴이 없는 프레임은 0.
- `y`: 프레임별 정수. `0=정상(눈 뜸)`, `1=눈감김·고개숙임`, `2=하품`, `3=불확실(얼굴 없음)` (`classes`)
- `fps`, `segments`(구간 라벨), `params`(사용한 임계값)도 함께 저장됩니다.

```python
d = json.load(open("labels_auto/SGA2100300S0001_자동라벨.json", encoding="utf-8"))
X, y = np.array(d["X"], dtype=np.float32), np.array(d["y"])   # (T, 11), (T,)
```

## 팀 규칙
- **임계값은 공유된 `best_params.json` 하나만 씁니다.** 각자 `calibrate`(또는 `all --manual-root`)를 돌리면 사람마다 `y` 기준이 달라집니다. 임계값을 다시 맞출 때는 한 사람이 손 라벨을 모아서 합니다.
- 클립은 겹치지 않게 나누고, 누가 어떤 클립을 했는지 기록합니다.
- 영상은 30fps 기준입니다. 다른 fps 영상이 섞이면 학습 전에 맞춥니다.
- `--roi`는 쓰지 않거나 모두 같은 값을 씁니다.
- 영상·라벨·`features/`·`labels_auto/`는 깃에 올리지 않습니다. Google Drive로 공유합니다.

## 현재 임계값 성능 (손 라벨 30클립, 0.5초 시점 460개)
| | 기본 임계값 | `best_params.json` |
| --- | --- | --- |
| 정확도 | 0.844 | 0.915 |
| macro F1 | 0.810 | 0.911 |

같은 클립으로 고르고 같은 클립으로 낸 점수라 낙관적입니다. 손 라벨 자체도 ±0.5초 오차가 있고, 결과는 사람 검수 전 초안입니다.
