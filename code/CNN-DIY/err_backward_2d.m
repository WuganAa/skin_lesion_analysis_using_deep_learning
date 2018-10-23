%In this function, we apply ReLU function for rectification
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
    err_backward_2d(err_matric,fd_matric,fd_weigh,fd_bias)
err_matric_size=size(err_matric);
%get the size of err_matric
w_size=size(fd_weigh);
%get the size of fd_weigh
weigh=zeros(w_size(1),w_size(2));
matric=zeros(err_matric_size(1)+w_size(1)-1,...
    err_matric_size(2)+w_size(2)-1);
fd_trans=Conv_Layer_2d(fd_matric,fd_weigh,fd_bias);
%apply Conv_Layer_2d function to get the forward result
b=0;
for i=1:err_matric_size(1)
    for j=1:err_matric_size(2)
        if fd_trans(i,j)~=0
            %consider whether forward result is zero or not, if the result
            %is zero, then the error can`t be accumlulated backward
            for k1=1:w_size(1)
                for k2=1:w_size(2)
                    matric(i+k1-1,j+k2-1)=matric(i+k1-1,j+k2-1)+...
                        fd_weigh(k1,k2)*err_matric(i,j);
                    weigh(k1,k2)=weigh(k1,k2)+...
                        fd_matric(i+k1-1,j+k2-1)*err_matric(i,j);
                    b=b+err_matric(i,j);
                    %apply complex derivative to support the backward error
                    %accumulation
                end
            end
        end
    end
end
end