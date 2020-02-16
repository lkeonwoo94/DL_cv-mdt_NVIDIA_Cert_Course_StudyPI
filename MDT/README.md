RNN은 말을 한게 긍정이다/부정이다 표현 할 수 있고

GAN은 input으로 noise를 준다.

* 1주차 segmentation    
* 2주차 word generation (예상하는 모델 학습)    
*('나는'이라고 입력주면 '지금'을 뱉고, '지금'을 넣으면 '랩실'을 뱉고, '랩실'을 넣으면 '있다'를 뱉고. '있다'를 넣으면 '끝'이라는 포인트)*     
* 3주차
이미지로 부터 세그멘테이션 했다 -> 특징을 추출햇다     
워드 제너레이션 -> 워드로 부터 특징을 추출함.    
2개를 결합해서, 영상을 주고 영상에 대한 이미지 캡셔닝을 함.    
이미지 캡셔닝이 되면 비디오 캡셔닝이 되죠.    


panoptic segmentation은 instance segmentation에서 하나 더 나아가서, 배경까지 segmentation 하는 것    
batch(=iter) & update(학습) ∈ epoch(shuffle)

---

실습

FCN
FC 대신 DeConv가 이썽서 뻥튀기가 되었다.
(공간정보를 살려서 가다가 upsampling을 하자.

fixme를 하나씩 수정해주세요. 
stride 2는 2x2를 의미해요

deconv로 256,256으로 올라갑니다.

opendata set이라고 medical dataset을 구할 수 있어요.


https://research.google/pubs/pub45732/
논문저널.
