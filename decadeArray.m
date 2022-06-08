function arr = decadeArray(startval, endval, stepmulti)
    i=startval;
    k = 1;

    arr = [];

    while i~=endval
       arr(k) = i;
       i = i * stepmulti;
       k = k + 1;
    end
     
end