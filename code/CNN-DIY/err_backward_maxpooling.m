%matric is the matrix before the err_matrix which accomodates the
%transmission error
%err_matric is the matrix which accomodates the errors
%fd_matric is the matrix in CNN net before the err_matrix

function matric=err_backward_maxpooling(err_matric,fd_matric)
err_matric_size=size(err_matric);
matric=zeros(err_matric_size(1)*2,err_matric_size(2)*2);
for i=1:err_matric_size(1)
    for j=1:err_matric_size(2)
        A=fd_matric(2*i-1:2*i,2*j-1:2*j);
        [row,col]=find(A==max(max(A)));
        matric(2*i-2+row,2*j-2+col)=err_matric(i,j);
    end
end
end