
function lala(a,b)
    return a+b
end


function lali(c,d)
    return (c*d + lala(c,d))
end


function type_check(x)
    @show(typeof(x))
end