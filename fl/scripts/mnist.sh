#!/bin/bash
# for j in 1 #2 3 4 5
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=0 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=200 --method='non-non-iid' --seed=40 --cnt=$j --defense='none' --local_epochs=1 &
#     sleep 5

#     # clients part 1
#     for ((i=1; i<=5; i++))
#     do
#         echo "Client $i is running"
#         CUDA_VISIBLE_DEVICES=0 python client.py --DevNum=$i --method='non-non-iid' --TotalDevNum=10 --seed=40 --defense='none' --attack='gs' &
#         sleep 5
#     done
#     echo "clients part 1 start"

#     # clients part 2
#     for ((i=6; i<=10; i++))
#     do
#         echo "Client $i is running"
#         CUDA_VISIBLE_DEVICES=0 python client.py --DevNum=$i --method='non-non-iid' --TotalDevNum=10 --seed=40 --defense='none' --attack='gs' &
#         sleep 5
#     done
#     echo "clients part 2 start"

#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"
#!/bin/bash

# for j in 1 
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=1 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=200 --method='non-iid' --seed=40 --cnt=$j --defense='dp' --local_epochs=1 &
#     sleep 5
#     # clients part 1
#     i=1
#     while [ $i -le 5 ]
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --DevNum=$i --method='non-iid' --TotalDevNum=10 --seed=40 --defense='dp' --scale=1e-2 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         i=$((i+1))
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while [ $i -le 10 ]
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --DevNum=$i --method='non-iid' --TotalDevNum=10 --seed=40 --defense='dp' --scale=1e-2 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         i=$((i+1))
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"


#!/bin/bash

# for j in 1 
# do
#     echo "Job $j is running"

#     # 启动服务器
#     CUDA_VISIBLE_DEVICES=1 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=200 --method='non-iid' --seed=40 --cnt=$j --defense='cp' --local_epochs=1 &
#     sleep 5

#     # 客户端部分 1
#     i=1
#     while [ $i -le 5 ]
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --DevNum=$i --method='non-iid' --TotalDevNum=10 --seed=40 --defense='cp' --percent_num=70 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         i=$((i+1))
#     done
#     echo "clients part 1 start"

#     # 客户端部分 2
#     i=6
#     while [ $i -le 10 ]
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --DevNum=$i --method='non-iid' --TotalDevNum=10 --seed=40 --defense='cp' --percent_num=70 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         i=$((i+1))
#     done
#     echo "clients part 2 start"

#     # 等待所有后台进程完成
#     wait
# done

# echo "all workers done"



# for j in 1 
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=1 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=200 --method='non-iid' --seed=40 --cnt=$j --defense='dcs' --local_epochs=1 &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --mixup --DevNum=$i --startpoint='none' --method='non-iid' --TotalDevNum=10 --seed=40 --defense='dcs' --percent_num=70 --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --mixup --DevNum=$i --startpoint='none' --method='non-iid' --TotalDevNum=10 --seed=40 --defense='dcs' --percent_num=70 --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"


#!/bin/bash

for j in 1 
do
    echo "Job $j is running"
    # server
    CUDA_VISIBLE_DEVICES=1 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=200 --method='non-iid' --seed=40 --cnt=$j --defense='soteria' --local_epochs=2 &
    sleep 5
    # clients part 1
    i=1
    while((i<=5))
    do
        CUDA_VISIBLE_DEVICES=1 python client.py --DevNum=$i --method='non-iid' --TotalDevNum=10 --seed=40 --defense='soteria' --percent_num=20 --layer_num=6 --attack='gs' &
        echo "Client $i is running"
        sleep 5
        let "i+=1"
    done
    echo "clients part 1 start"
    # clients part 2
    i=6
    while((i<=10))
    do
        CUDA_VISIBLE_DEVICES=1 python client.py --DevNum=$i --method='non-iid' --TotalDevNum=10 --seed=40 --defense='soteria' --percent_num=20 --layer_num=6 --attack='gs' &
        echo "Client $i is running"
        sleep 5
        let "i+=1"
    done
    echo "clients part 2 start"
    # wait for current cnt to finish
    wait
done
echo "all workers done"


# for j in 1 2 3 4 5
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=1 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=200 --method='non-iid' --seed=40 --cnt=$j --defense='dcs_cp' --local_epochs=1 &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --mixup --DevNum=$i --startpoint='none' --method='non-iid' --TotalDevNum=10 --seed=40 --defense='dcs_cp' --percent_num=70 --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --mixup --DevNum=$i --startpoint='none' --method='non-iid' --TotalDevNum=10 --seed=40 --defense='dcs_cp' --percent_num=70 --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"

# for j in 1 2 # 3 4 5
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=0 python server.py --output_dir='./logs/lambda_y/1.0' --minfit=10 --mineval=10 --minavl=10 --num_rounds=200 --method='non-iid' --seed=40 --cnt=$j --defense='dcs' --local_epochs=1 &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=0 python client.py --output_dir='./logs/lambda_y/1.0' --lambda_y=1.0 --mixup --DevNum=$i --startpoint='none' --method='non-iid' --TotalDevNum=10 --seed=40 --defense='dcs' --percent_num=70 --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --xsim_thr=300. --epsilon=0.01 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=0 python client.py --output_dir='./logs/lambda_y/1.0' --lambda_y=1.0 --mixup --DevNum=$i --startpoint='none' --method='non-iid' --TotalDevNum=10 --seed=40 --defense='dcs' --percent_num=70 --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --xsim_thr=300. --epsilon=0.01 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"




