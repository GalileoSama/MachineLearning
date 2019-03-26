def hanshu1(in_money, exchange_rate):
    ##汇率兑换函数
    out=in_money*exchange_rate
    return out

rate = 6.77
money = input("请输入金额(usd/rmb）：")
unit = money[-3:]
if unit=="usd":
   exchange_rate= rate
elif unit =="rmb":
    exchange_rate = 1/rate
else:
    exchange_rate = -1

if exchange_rate !=-1:
    in_money = eval(money[:-3])
    #调用函数
    out_money = hanshu1(in_money, exchange_rate)
    print("转换后的金额",out_money)
else:
    print("暂不支持！")