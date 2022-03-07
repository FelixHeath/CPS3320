
#AS2
#wangzhitao 1098487
#xiangtingfeng 1098608

number = int(input('please input an integer：')) # input the number
    # start of while loop
while number <= 0: # check positive or negative
    number = int(input("please input a new integer which is positive")) # reinput if negative
        #end of while loop
    #start of if else
if number % 2 ==0 and number % 3 ==0 and number % 4 ==0: # check multiple 2,3,4
    print('The number', number, 'is multiple of 2,3,4') # output is true
else:
    print('The number', number,'is not a multiple of 2,3,4') # output is false
print('End Of Program') # end of program, end of if else
