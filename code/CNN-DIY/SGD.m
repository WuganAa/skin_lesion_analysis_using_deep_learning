%in the function, wi and bi refers to the relative weighs and bias
%img is the image matrix
%label is the image label
function [w1,w2,w3,w4,w5,w6,w7,b1,b2,b3,b4,b5,b6,b7]=...
    SGD(img,label,a,w1,w2,w3,w4,w5,w6,w7,b1,b2,b3,b4,b5,b6,b7)
img_1=Conv_Layer_3d(img,w1,b1);
img_2=max_pooling(img_1);
img_3=Conv_Layer_2d(img_2,w2,b2);
img_4=max_pooling(img_3);
img_5=Conv_Layer_2d(img_4,w3,b3);
img_6=max_pooling(img_5);
img_7=Conv_Layer_2d(img_6,w4,b4);
img_8=max_pooling(img_7);
img_9=Conv_Layer_2d(img_8,w5,b5);
img_10=max_pooling(img_9);
img_11=Conv_Layer_2d(img_10,w6,b6);
result=softmax_layer(img_11,w7,b7);
%construt the CNN net
err=label-result;
[img_11_err,w7_err,b7_err]=err_backward_2d_softmax(err*a,img_11,w7,b7);
[img_10_err,w6_err,b6_err]=err_backward_2d(img_11_err,img_10,w6,b6);
img_9_err=err_backward_maxpooling(img_10_err,img_9);
[img_8_err,w5_err,b5_err]=err_backward_2d(img_9_err,img_8,w5,b5);
img_7_err=err_backward_maxpooling(img_8_err,img_7);
[img_6_err,w4_err,b4_err]=err_backward_2d(img_7_err,img_6,w4,b4);
img_5_err=err_backward_maxpooling(img_6_err,img_5);
[img_4_err,w3_err,b3_err]=err_backward_2d(img_5_err,img_4,w3,b3);
img_3_err=err_backward_maxpooling(img_4_err,img_3);
[img_2_err,w2_err,b2_err]=err_backward_2d(img_3_err,img_2,w2,b2);
img_1_err=err_backward_maxpooling(img_2_err,img_1);
[w1_err,b1_err]=err_backward_3d_easy(img_1_err,img,w1,b1);
%get the backward error by err_backward function
w1=w1+w1_err;
w2=w2+w2_err;
w3=w3+w3_err;
w4=w4+w4_err;
w5=w5+w5_err;
w6=w6+w6_err;
w7=w7+w7_err;
b1=b1+b1_err;
b2=b2+b2_err;
b3=b3+b3_err;
b4=b4+b4_err;
b5=b5+b5_err;
b6=b6+b6_err;
b7=b7+b7_err;
%modify the weighs and biases according to the backward errors
end