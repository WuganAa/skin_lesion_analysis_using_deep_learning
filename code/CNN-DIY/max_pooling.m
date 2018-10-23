%apply max_pooling to dispose the image
%in this function, the max_pooling function is applied on an area of 2*2
%and the step is 2
%img is the image inputed
function y=max_pooling(img)
img_size=size(img);
new_img=zeros(img_size(1)/2,img_size(2)/2);
for i=1:img_size(1)/2
    for j=1:img_size(2)/2
        new_img(i,j)=max(max(img(2*i-1:2*i,2*j-1:2*j)));
    end
end
y=new_img;
end