import datetime
import pytz

def write_time_to_file(label):
    """
    Append the current time with a label to a file, using JST timezone.
    """
    jst = pytz.timezone('Asia/Tokyo')
    current_time = datetime.datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S")
    
    with open("/home/ryo/scrape/PBL_stock/time_log.txt", "a") as file:
        file.write(f"{label}: {current_time}\n")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2 or sys.argv[1] not in ["start", "end"]:
        print("Usage: python log_time.py [start|end]")
        sys.exit(1)

    write_time_to_file("Script Started" if sys.argv[1] == "start" else "Script Ended")

