function res = phiMaker(x1,x2)
    
   res = ones(length(x1),1);
    for t=1:6
        for p=flip(0:t)
            res = [res, x1.^p .* x2.^(t-p)];
        end
    end
end

