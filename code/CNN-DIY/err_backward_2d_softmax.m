%In this function, we apply sigmoid function for rectification
%err_matric is the matrix which accomodates the errors
%fd_matric is the matrix in CNN net before the err_matrix
%fd_weigh is the weigh matrix before the err_matrix
%fd_bias is the bias before the err_matrix
%matric is the matrix before the err_matrix which accomodates the
%transmission error
%weigh is the err_weigh before the err_matrix which accomodates the
%transmission error
%bias is the err_bias before the err_matrix which accomodates the
%transmission error
function [matric,weigh,b]=...
    err_backward_2d_softmax(err_matric,fd_matric,fd_weigh,fd_bias)
err_matric_size=size(err_matric);
%get the size of err_matric
w_size=size(fd_weigh);
%get the size of fd_weigh
weigh=zeros(w_size(1),w_size(2));
matric=zeros(err_matric_size(1)+...
    w_size(1)-1,err_matric_size(2)+w_size(2)-1);
fd_trans=softmax_layer(fd_matric,fd_weigh,fd_bias);
b=0;
for i=1:err_matric_size(1)
    for j=1:err_matric_size(2)
        for k1=1:w_size(1)
            for k2=1:w_size(2)
                matric(i+k1-1,j+k2-1)=matric(i+k1-1,j+k2-1)+...
                    fd_weigh(k1,k2)*err_matric(i,j)*...
                    fd_trans(i,j)*(1-fd_trans(i,j));
                weigh(k1,k2)=weigh(k1,k2)+fd_matric(i+k1-1,j+k2-1)*...
                    err_matric(i,j)*fd_trans(i,j)*(1-fd_trans(i,j));
                b=b+err_matric(i,j)*fd_trans(i,j)*(1-fd_trans(i,j));
                %apply complex derivative to support the backward error
                %accumulation
            end
        end
    end
end
end