import codecs

def nums(k):
    num=[]
    while True:
        num.append(k%10)
        k=k//10
        if k==0:
            break
    num.reverse()
    return(num)

def super_script(k):
    pows=''
    num=nums(k)
    for i in num:
        if i>1 and i<=3:
            pows+=r'\u00b{}'.format(i)
        if i==1:
            pows+=r'\u00b{}'.format(9)
        if i>3 or i==0:
            pows+=r'\u207{}'.format(i)
    return codecs.decode(pows, 'unicode_escape')

def sub_script(k):    
    pows=''
    num=nums(k)
    for i in num:
        pows+=r'\u208{}'.format(i)
    return codecs.decode(pows, 'unicode_escape')

if __name__=='__main__':
    print(super_script(1))
    print(sub_script(1))
