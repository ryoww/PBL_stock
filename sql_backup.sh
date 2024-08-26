#!/bin/bash

BACKUP_PATH="/mnt/main/Backup/SQL/stock_backup.sql"

DB_USER="stock"
DB_PASSWORD="ryotaro1212"
DB_NAME="stock"

mysqldump -u $DB_USER -p $DB_PASSWORD > $BACKUP_PATH

