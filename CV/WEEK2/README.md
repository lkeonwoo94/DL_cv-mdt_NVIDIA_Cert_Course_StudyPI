> **overfitting**

* 데이터가 많은 경우: **class들간에 각각의 데이터 양이 일정하게 비슷해야**만 overfitting을 피할 수 있습니다.
overfitting이 발생한다면, 많은 수의 data를 포함하는 class에 치우쳐져서 학습되기 때문입니다.    


* 데이터가 적은 경우: 마찬가지로 overfitting이 발생할 수 있습니다. 1장에 대해서는 확실하게 학습을 할 수는 있지만, 그 1장과 조금만 다른 영상이지만 같은 종류의 게가 입력으로 온다면 틀릴 가능성이 높습니다. **최소 클래스별 500개**정도씩 가지고 계시다면 시작해볼 순 있을 것 같습니다.

---
8:00~8:30  (이론)    

> ## **overfitting**
* little overfitting : validation과 test가 비슷한 수치로 상승 => **model을 더 깊게 만들어서 해결**
* strong overfitting : valdiation과 test가 벌어짐 => Regularization, Early stopping, Dropout 등으로 해결

> ## **deployment**
학습된 모델을 porting 하는 것.

> ## **Transferlearning**


---
8:30~ 실습
* 1. OPEN DIGITS 클릭
* 2. New classification dataset . 

1. GPU TASK2 : new dataset 생성
2. GPU TASK3 : deployment , 전처리는 resize

snapshot interval..?

[캐글](https://www.kaggle.com/)
