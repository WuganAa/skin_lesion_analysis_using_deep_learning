function result=CNN_net_2(img,w1,w2,w3,w4,w5,w6,b1,b2,b3,b4,b5,b6)
img_1=Conv_Layer_3d(img,w1,b1);
img_2=max_pooling(img_1);
img_3=Conv_Layer_2d(img_2,w2,b2);
img_4=max_pooling(img_3);
img_5=Conv_Layer_2d(img_4,w3,b3);
img_6=max_pooling(img_5);
img_7=Conv_Layer_2d(img_6,w4,b4);
img_8=max_pooling(img_7);
img_9=Conv_Layer_2d(img_8,w5,b5);
result=softmax_layer(img_9,w6,b6);
end