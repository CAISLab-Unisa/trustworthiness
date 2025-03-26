function [] = store_create_path(store_path)
%STORE_CREATE_PATH Summary of this function goes here
%  Create a Path if does not exists
  if ~exist(store_path, 'dir')
       mkdir(store_path)
  end
  
end

