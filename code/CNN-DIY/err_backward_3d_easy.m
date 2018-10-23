%In this function, we apply ReLU function for rectification
%err_matric is the matrix which accomodates the errors
%fd_matric is the matrix in CNN net before the err_matrix
%fd_weigh is the weigh matrix before the err_matrix
%fd_bias is the bias before the err_matrix
%weigh is the err_weigh before the err_matrix which accomodates the
%transmission error
%bias is the err_bias before the err_matrix which accomodates the
%transmission error
function [weigh,b]=...
    err_backward_3d_easy(err_matric,fd_matric,fd_weigh,fd_bias)
err_matric_size=size(err_matric);
%get the size of err_matric
if length(err_matric_size)==2
    err_matric_size(3)=1;
end
w_size=size(fd_weigh);
%get the size of fd_weigh
weigh=zeros(w_size(1),w_size(2),w_size(3));
fd_trans=Conv_Layer_3d(fd_matric,fd_weigh,fd_bias);
%apply Conv_Layer_3d function to get the forward result
b=0;
for i=1:err_matric_size(1)
    for j=1:err_matric_size(2)
        for k=1:err_matric_size(3)
            if fd_trans(i,j,k)~=0
                %consider whether forward result is zero or not, if the 
                %result is zero, then the error can`t be accumlulated 
                %backward
                for k1=1:w_size(1)
                    for k2=1:w_size(2)
                        for k3=1:w_size(3)
                            weigh(k1,k2,k3)=weigh(k1,k2,k3)+...
                                fd_matric(i+k1-1,j+k2-1,k+k3-1)*...
                                err_matric(i,j,k);
                            b=b+err_matric(i,j,k);
                            %apply complex derivative to support the 
                            %backward error accumulation
                        end
                    end
                end
            end
        end
    end
end
end