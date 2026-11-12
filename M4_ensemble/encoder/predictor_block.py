class Predictor:
    """발화조각을 하나씩 받아 환자의 증상을 예측하는 모델 래퍼.

    모델: KLUE-RoBERTa-base 누적 입력 증상 분류기 (encoder_model.py)
      - 매 발화마다 지금까지 받은 발화 1..t만 이어 붙여 9개 증상 확률을 낸다.
      - 대화의 전체 길이(총 발화 수, 총 통화 시간)와 오디오는 쓰지 않는다.
      - 미션별 가중치: 4-1은 BCE + soft Jaccard로 학습한 모델(애매해도 가장 그럴듯한 증상을 확신),
        4-2는 BCE로 학습한 모델(확률이 정직해서 중단 시점 판단에 유리). 둘 다 체크포인트 파일 하나에 들어 있다.

    판정 규칙 (운영사무국 안내: Decision Threshold 0.5 고정)
      - Mission 4-1 : 확률 0.5 이상인 증상을 매 시점 반환한다. 없으면 빈 리스트.
      - Mission 4-2 : 확률 0.5 이상인 증상이 있고, 그 증상 집합이 k번 연속 같으며,
                      지금까지 받은 발화 수가 t_min 이상이면 중단하고 그 집합을 반환한다.
                      t_min, k는 팀 자체 검증셋(Train 분할)에서 고른 값으로 체크포인트에 저장돼 있다.
                      끝까지 중단하지 않으면 마지막 시점의 0.5 이상 증상을 반환한다.
    """

    THR = 0.5   # 판정 기준과 중단 판단 기준 (규정상 고정)

    def __init__(self, ckpt_path: str, mission: str, device: str = "cpu") -> None:
        self.ckpt_path = ckpt_path
        self.mission = mission          # "4-1" 또는 "4-2"
        self.device = device
        self.model = self.load_model(ckpt_path)

    def load_model(self, ckpt_path: str):
        """체크포인트 1개(가중치 + 설정 + 토크나이저)를 인터넷 없이 로드한다."""
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)                    # 다른 폴더에서 실행해도 모델 코드를 찾도록
        from encoder_model import StreamingEncoder      # inference.py와 같은 폴더의 모델 코드
        model = StreamingEncoder.load(ckpt_path, device=self.device, mission=self.mission)
        self.t_min = model.halt_t_min                   # 4-2: 이 발화 수부터 중단 가능
        self.k = model.halt_k                           # 4-2: 같은 예측이 몇 번 연속이면 중단
        return model

    def reset(self) -> None:
        """새 통화 시작: 받은 발화와 4-2 상태를 모두 비운다."""
        self.model.reset()
        self.prev = None        # 직전 시점의 0.5 이상 증상 집합
        self.run = 0            # 같은 집합이 연속으로 나온 횟수 (빈 예측이면 0)
        self.last = []          # 가장 최근 시점의 판정 (finalize용)

    def step(self, utt: Utterance) -> Optional[List[str]]:
        """발화 1개를 추가하고, 지금까지의 발화만으로 판정한다 (텍스트와 화자만 사용)."""
        probs = self.model.add(utt.speaker, utt.text)
        pred = [c for c, p in zip(SYMPTOM_CLASSES, probs) if p >= self.THR]
        if self.mission == "4-1":
            return pred

        # Mission 4-2: 중단 판단
        key = tuple(pred)
        if pred:
            self.run = self.run + 1 if key == self.prev else 1
        else:
            self.run = 0
        self.prev, self.last = key, pred
        if self.model.n_seen >= self.t_min and self.run >= self.k:
            return pred         # 중단: 이 시점의 0.5 이상 증상
        return None             # 더 듣기

    def finalize(self) -> List[str]:
        """4-2에서 끝까지 중단하지 않은 경우: 마지막 시점의 0.5 이상 증상."""
        return list(self.last)
