# Transfer Learning

트랜스퍼 러닝을 하지 않으면 스크래치 러닝.

둘으 ㅣ차이는

---
freeze 라는게 , 

--- 
앞20분다시듣기
트랜스퍼 러닝은 데이터가 많으면 필요가 없다. 결국에는 converge가 된다.
---
8시30~ 9시 : obj detect
1. sliding window : 
2. NMS : 1개의 box만 추출
3. HOG : 미분값들의 차이 
4. DPM: HOG + SVM

5. SS

OBJ 공개 데이터 셋 : ~~~~ 등등..

앵커 : 피쳐맵을 뽑기 위해 다양한 필터를 사용한다.

레티나넷 : 피쳐맵이 작아지수록 찾아내는 피쳐의 크기가 커진다.
---
실습 task5

filter size 만큼 뜯어서 cnn 넣고 classifier 돌리면, size에 대해서 이것이 무엇인지에 대한 확률(confidence 값)이 hitmap으로 나옴.
확률이 높은 영역을 box를 치면, detection이 됨. 

stride를 dense하게.

> 두번째 실습 방법
Fully Connected Layer 사용. (FC = Flatten)

Deconv / Upsampling으로 Output을 키운다( FC대신 Fully convolution.

> 세번째 접근법 (DetectNet : NVIDIA에서 개발한 방법)
feature의 위치정보도 나와있음. detectuib 알고리즘은 **좌표정보**도 필요. 좌표정보도 loss로 들어가서 학습이 돼야함.

