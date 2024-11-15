#!/bin/bash

BACKUP_PATH="/mnt/main/Backup/SQL/stock_backup.sql"

DB_USER="stock"
DB_PASSWORD="ryotaro1212"
DB_NAME="stock"

/usr/bin/mysqldump -u $DB_USER -p$DB_PASSWORD $DB_NAME > $BACKUP_PATH

