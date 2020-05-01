from julia import Julia
jul = Julia(compiled_modules=False)

# ds = """
# function lala(a,b)
#     return a+b
# end
#
# """

# x = jul.include("test.jl")


jul.eval('include("test.jl")')


lala = jul.eval('lala')

lali = jul.eval('lali')

# y = jul.eval(ds)

# y = jul.eval("test.jl")


type_check = jul.eval('type_check')

# print(lala)
# print(lali)
#
# print(lala(1,2))
# print(lali(1,2))


x = [[1,2,3], [1,2,3]]
type_check(x)
