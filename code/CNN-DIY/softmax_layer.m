%img is the feature image inputed
%wis the weigh of the convolutional kernel, b is the bias
function y=softmax_layer(img,w,b)
img_size=size(img);
weigh_size=size(w);
new_img=zeros(img_size(1)-(weigh_size(1)-1),...
    img_size(2)-(weigh_size(2)-1));
for i=1:img_size(1)-(weigh_size(1)-1)
    for j=1:img_size(2)-(weigh_size(2)-1)
        new_img(i,j)=sum(sum(w.*img(i:i+weigh_size(1)-1,...
            j:j+weigh_size(2)-1)))+b;
    end
end
y=(1+exp(-new_img)).^(-1);
%apply sigmoid function to map the result to a certain value between 0 and
%1
end