%img is the image inputed
%w is the kernel of the convolutional calculation, b is the bias
function y=Conv_Layer_3d(img,w,b)
img_size=size(img);
weigh_size=size(w);
new_img=zeros(img_size(1)-(weigh_size(1)-1),...
    img_size(2)-(weigh_size(2)-1),...
    img_size(3)-(weigh_size(3)-1));
for i=1:img_size(1)-(weigh_size(1)-1)
    for j=1:img_size(2)-(weigh_size(2)-1)
        for k=1:img_size(3)-(weigh_size(3)-1)
            new_img(i,j,k)=sum(sum(sum(w.*img(i:i+weigh_size(1)-1,...
                j:j+weigh_size(2)-1,k:k+weigh_size(3)-1))))+b;
            %apply convolutional calculation
        end
    end
end
y=max(0,new_img);
%apply ReLU function
end