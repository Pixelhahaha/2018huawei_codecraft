# coding=utf-8
from __future__ import division
import sys
import os
import predictor
import datetime
import copy
import math
start = datetime.datetime.now()

def main():
    print 'main function begin.'
    if len(sys.argv) != 4:
        print 'parameter is incorrect!'
        print 'Usage: python esc.py ecsDataPath inputFilePath resultFilePath'
        exit(1)
    # Read the input files
    ecsDataPath = sys.argv[1]
    inputFilePath = sys.argv[2]
    resultFilePath = sys.argv[3]

    ecs_infor_array = read_lines(ecsDataPath)
    input_file_array = read_lines(inputFilePath)


    # implementation the function predictVm
    # predic_result=input_file_array
    predic_result,input_data,num_ = predictor.predict_vm(ecs_infor_array, input_file_array)


    # write the result to output file

    if len(predic_result) != 0:
        write_result(predic_result, resultFilePath,input_data,num_)
    else:
        predic_result.append("NA")
        write_result(predic_result, resultFilePath,input_data)
    print 'main function end.'

    end = datetime.datetime.now()
    print (end - start)

def write_result(array, outpuFilePath,input_data,num_):
    num_xuniji = 0
    for item in array[0]:
        num_xuniji+=item[1]
    num_xuniji2 = 0
    for i in range(len(array[1])):
        for k in range(len(array[1][i])):
            num_xuniji2 += array[1][i][k][1]
    if num_xuniji != num_xuniji2:
        return 0
    with open(outpuFilePath, 'w') as output_file:
        output_file.write("%s\n" % num_xuniji) # 预测的虚拟机总数
        for item in array[0]:
            STR = 'flavor'+str(item[0])+' '+str(item[1])+'\n'  #虚拟机规格名称1 虚拟机个数
            output_file.write(STR)

        for k in range(input_data[0]):
            h = input_data[k+1][0][0]
            if h == 'G':
                output_file.write("\n")  # 空行
                STR0 = input_data[k + 1][0] + ' ' + str(num_[0])
                output_file.write("%s\n" % STR0)  # 所需物理服务器总数
                for i in range(num_[0]):
                    STR = input_data[k+1][0] + '-' + str(i + 1)
                    for ii in range(len(array[1][i])):
                        # for iii in range(len(array[1][i][ii])):
                        STR += ' flavor' + str(array[1][i][ii][0]) + ' ' + str(array[1][i][ii][1])
                    STR += '\n'
                    output_file.write(STR)  # 物理服务器1 虚拟机规格名称1 能放置该类型虚拟机个数 虚拟机规格名称2 能放置该类型虚拟机个数 ……

            if h == 'L':
                output_file.write("\n")  # 空行
                STR0 = input_data[k + 1][0] + ' ' + str(num_[1])
                output_file.write("%s\n" % STR0)  # 所需物理服务器总数
                for i in range(num_[1]):
                    STR = input_data[k+1][0] + '-' + str(i + 1)
                    for ii in range(len(array[1][i + num_[0] ])):
                        # for iii in range(len(array[1][i][ii])):
                        STR += ' flavor' + str(array[1][i + num_[0] ][ii][0]) + ' ' + str(array[1][i+ num_[0]][ii][1])
                    STR += '\n'
                    output_file.write(STR)  # 物理服务器1 虚拟机规格名称1 能放置该类型虚拟机个数 虚拟机规格名称2 能放置该类型虚拟机个数 ……


            if h == 'H':
                output_file.write("\n")  # 空行
                STR0 = input_data[k + 1][0] + ' ' + str(num_[2])
                output_file.write("%s\n" % STR0)  # 所需物理服务器总数
                for i in range(num_[2]):
                    STR = input_data[k+1][0] + '-' + str(i + 1)
                    for ii in range(len(array[1][i+ num_[0]+ num_[1]])):
                        # for iii in range(len(array[1][i][ii])):
                        STR += ' flavor' + str(array[1][i + num_[0]+ num_[1] ][ii][0]) + ' ' + str(array[1][i+ num_[0]+ num_[1] ][ii][1])
                    STR += '\n'
                    output_file.write(STR)  # 物理服务器1 虚拟机规格名称1 能放置该类型虚拟机个数 虚拟机规格名称2 能放置该类型虚拟机个数 ……



def read_lines(file_path):
    if os.path.exists(file_path):
        array = []
        with open(file_path, 'r') as lines:
            for line in lines:
                array.append(line)
        return array
    else:
        print 'file not exist: ' + file_path
        return None


if __name__ == "__main__":
    main()