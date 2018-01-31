import sys
import os
import re
import pdb
import xlwt
 
help_ = '''
Usage:
    python tpi.py log.txt
'''

#data_list= {}
data_list1= []
data_list2= []
cnt=0
times=0.0
table_val=''
name_list1= ['allocate','run','configure','tensor_copy','ACL_CONV','ACL_FC','ACL_LRN','ACL_POOLING','ACL_RELU','ACL_SOFTMAX']




def getvalpairs(words):
    name=''
    val=''
    for word in words:
        if word=='':
            continue
        if name=='':
            name=word
        else:
            val=word
            break;
        #print word,
    #print ''
    return (name,val)

def addpairstolist(db,name,val,idx):
     #pdb.set_trace()
     #if idx in db:
     #   db[idx]['val'] += val
     #else:
     #   db[idx] = {'val':val,'name':name}

     #pdb.set_trace()

     for i in db:
         if i['name']==name:
             i['val'] += val
             return
     db.append({'idx':idx,'val':val,'name':name})

def gettabnum(line):
    start=line.find(':')
    if start==-1:
        start=0
    else:
        start+=1
    #pdb.set_trace()
    str=line[start:-1].lstrip(' ')
    words=re.split('\t',str)
    idx=0
    for word in words:
        idx+=1
        if word=='':
            continue
        break
    return idx

def decodefile(logfile):
    data_list=data_list1
    for line in open(logfile):
        if line.find(':')==-1:
            continue
        #pdb.set_trace()
        #print line,
        idx=gettabnum(line)
        words=re.split('\t|:| |\r|\n',line)
        #print(words)
        (name,val)=getvalpairs(words)
        #print (name,float(val),eval(val))
        if name == 'used' and val == 'time':
            data_list=data_list2
        try:
            addpairstolist(data_list,name,float(val),idx)
        except ValueError as e:
            #print(line)
            continue

def printresult(db):
    #for i in db:
    #    print i, db[i]['idx'],db[i]['val']
    #pdb.set_trace()
    db.sort(key=lambda obj:obj.get('idx'), reverse=False)
    tpi_start=0
    conv_str='ACL_CONV'
    find_acl = 0
    name_index=0
    global trow
    global tcol
    for i in db:
        if i['name']==conv_str:
            tpi_start=i['idx']

    tpi=0
    for i in db:
        if i['idx']>=tpi_start:
            tpi+=i['val']

    start=len('ACL_')

    table_head='TPI'+'\t'
    table_val='%.4f' % (tpi/times)+'\t'

    for i in db:
        #print i
        if i['idx']<tpi_start:
            if i['name'].find('ACL_')==0:
               table_head+=i['name'][start:]+'\t'
            else:
                table_head+=i['name']+'\t'
            table_val+='%.4f' % (i['val']/times)+'\t'

    print(table_head)
    print(table_val)

    table_head='TPI'+'\t'
    table_val='%.4f' % (tpi/times)+'\t'

    for i in db:
        if i['idx']>=tpi_start:
            if i['name'].find('ACL_')==0:
               #pdb.set_trace()
               table_head+=i['name'][start:]+'\t'
            else:
                table_head+=i['name']+'\t'
            table_val+='%.4f' % (i['val']/times)+'\t'

    print(table_head)
    print(table_val)

    ws.write(trow, tcol, 'TPI')
    ws.write(trow+1,tcol,'%.4f' % (tpi/times))
    tcol+=1

    temp_row=trow
    temp_col=tcol
    for i in name_list1:
        if i.find('ACL_')==0 and find_acl==0:
            temp_row+=2
            temp_col=2
            find_acl=1
        ws.write(temp_row,temp_col,i)
        ws.write(temp_row+1,temp_col,'0')
        temp_col+=1
    find_acl=0

    for i in db:
        curname=i['name']
        curvalue='%.4f' % (i['val']/times)
        if curname == 'ACL_BN':
            ws.write(trow+2,7,curname)
            ws.write(trow+3,7,curvalue)

        if curname in name_list1:
            val_col=name_list1.index(curname)+2
            val_row=trow
            # print ('name found'+ curname + curvalue)
            # print(val_col)
            # print (val_row)
            if val_col>5:
                val_col-=4
                val_row+=2
            ws.write(val_row,val_col,curname)
            ws.write(val_row+1,val_col,curvalue)

    tcol=0
    trow+=4


if __name__ == '__main__' :
    if len(sys.argv) < 2:
        print(help_)
        sys.exit()
    else:
        logfile = sys.argv[1]

    filename = os.path.basename(logfile)
    decodefile(logfile)

    wb = xlwt.Workbook()
    ws = wb.add_sheet('testsheet',True)
    trow = 0
    tcol = 0
    cnt=1
    times=1.0
    table_val=''
    print('1st time:')
    ws.write(trow,tcol,'1st time')
    tcol+=1
    printresult(data_list1)

    cnt=2
    times=10.0
    table_val=''
    print('\nAverage of 2-11 times:')
    ws.write(trow, tcol, '2-11 times')
    tcol+=1
    printresult(data_list2)
    wb.save(filename+'.xls')
    print ('Xls file generated')


