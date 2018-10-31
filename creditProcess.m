function [M, A, B] = creditProcess()

% preprocess the credit data. The output of the function is the centered
% data as matrix M. Centered low educated group A and high educated as
% group B. 

addpath data/credit

data = csvread('default_degree.csv', 2, 1);

% vector of sensitive attribute.
sensitive = data(:,1);

% normalizing the sensitive attribute vetor to have 0 for grad school and 
% university level education and positive value for high school, other
normalized = (sensitive-1).*(sensitive-2);

% getting rid of the colum corresponding to the senstive attribute.
data = data(:,2:22);

n = size(data, 2);

% centering the data and normalizing the variance across each column
for i=1:n
   data(:,i) = data(:,i) - mean(data(:,i));
   data(:,i) = data(:,i)/std(data(:,i));
end

% data for low educated populattion
data_lowEd = data(find(normalized),:);
lowEd_copy = data_lowEd;

% date for high educated population
data_highEd = data(find(~normalized),:);
highEd_copy = data_highEd;

mean_lowEd = mean(lowEd_copy,1);
mean_highEd = mean(highEd_copy, 1);

% centering data for high- and low-educated
for i=1:n
   lowEd_copy(i,:) = lowEd_copy(i,:) - mean_lowEd;
end

for i=1:n
   highEd_copy(i,:) = highEd_copy(i,:) - mean_highEd;
end


M = data;
A = lowEd_copy;
B = highEd_copy;
    

end
