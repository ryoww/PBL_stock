#!/bin/bash

BACKUP_PATH="/mnt/main/Backup/SQL/stock_backup.sql"

DB_USER="ryo"
DB_PASSWORD="ryotaro1212"
DB_NAME="stock"

/usr/bin/mysqldump -u $DB_USER -p --default-character-set=utf8mb4 --routines --triggers --events --hex-blob $DB_PASSWORD $DB_NAME > $BACKUP_PATH

