from julia import Julia
jul = Julia(compiled_modules=False)

ds = """
function lala(a,b)
    return a+b
end

"""





x = jul.include("test.jl")
# y = jul.eval(ds)

# y = jul.eval("test.jl")
print(x(1,2))
