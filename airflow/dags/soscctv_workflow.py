from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'simyo',
    'start_date': datetime(2026, 3, 20),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'SOS_CCTV_ML_Pipeline',
    default_args=default_args,
    schedule='@weekly', # 매주 일요일 실행
    catchup=False,
    tags=['soscctv', 'mlops']
) as dag:

    # 1단계: 오늘 날짜 폴더 생성 후 new 데이터를 날짜별로 이사시키기
    collect_data = BashOperator(
        task_id='collect_new_data',
        bash_command="""
            TARGET_DATE=$(date +%Y-%m-%d)
            TARGET_DIR="/Users/hyojin/Desktop/SOScctv/data/processed/$TARGET_DATE"
            
            mkdir -p "$TARGET_DIR"
            
            COUNT=$(ls /Users/hyojin/Desktop/SOScctv/data/new/*.npy 2>/dev/null | wc -l)
            
            if [ $COUNT -gt 0 ]; then
                echo "🚚 $COUNT 개의 데이터를 $TARGET_DIR 로 이동합니다."
                mv /Users/hyojin/Desktop/SOScctv/data/new/*.npy "$TARGET_DIR/"
            else
                echo "ℹ️ 이동할 새 데이터가 없습니다."
            fi
        """
    )

    # 2단계: 전체 데이터 개수 확인 (하위 폴더 포함)
    check_data = BashOperator(
        task_id='check_total_data',
        bash_command="find /Users/hyojin/Desktop/SOScctv/data/processed -name '*.npy' | wc -l"
    )

    # 3단계: 모델 재학습 실행
    train_model = BashOperator(
        task_id='retrain_action_model',
        bash_command="/Users/hyojin/Desktop/SOScctv/venv/bin/python /Users/hyojin/Desktop/SOScctv/scripts/train.py"
    )

    # 파이프라인 순서 정의
    collect_data >> check_data >> train_model
