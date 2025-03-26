function [] = store_create_path(store_path)
%STORE_CREATE_PATH 

  if ~exist(store_path, 'dir')
       mkdir(store_path)
  end
  
end

