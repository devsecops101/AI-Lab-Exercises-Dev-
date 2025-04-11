#!/bin/bash
apt-get update
apt-get install -y docker.io docker-compose wget net-tools curl jq

# Setup directories
mkdir -p /opt/guacamole/init
cd /opt/guacamole

# Download SQL files
wget -q https://raw.githubusercontent.com/apache/guacamole-client/master/extensions/guacamole-auth-jdbc/modules/guacamole-auth-jdbc-postgresql/schema/001-create-schema.sql -O /opt/guacamole/init/001-create-schema.sql
wget -q https://raw.githubusercontent.com/apache/guacamole-client/master/extensions/guacamole-auth-jdbc/modules/guacamole-auth-jdbc-postgresql/schema/002-create-admin-user.sql -O /opt/guacamole/init/002-create-admin-user.sql

# Create connection setup SQL
cat << 'CONNECTIONS' > /opt/guacamole/init/003-create-connections.sql
INSERT INTO guacamole_connection_group (connection_group_name, type) VALUES ('Student VMs', 'ORGANIZATIONAL');
%{ for idx in range(var.student_count) }
INSERT INTO guacamole_connection (connection_name, protocol, max_connections, max_connections_per_user)
VALUES ('Student-VM-${idx + 1}', 'ssh', 40, 30);
DO $$
DECLARE
    connection_id integer;
    group_id integer;
BEGIN
    SELECT connection_group_id INTO group_id FROM guacamole_connection_group WHERE connection_group_name = 'Student VMs';
    SELECT last_value INTO connection_id FROM guacamole_connection_connection_id_seq;
    INSERT INTO guacamole_connection_parameter (connection_id, parameter_name, parameter_value)
    VALUES
    (connection_id, 'hostname', '${azurerm_linux_virtual_machine.student_vm[idx].private_ip_address}'),
    (connection_id, 'port', '22'),
    (connection_id, 'username', '${var.vm_admin_username}'),
    (connection_id, 'private-key', '${replace(trimspace(tls_private_key.ssh_key.private_key_pem), "'", "''")}');
    INSERT INTO guacamole_connection_permission (entity_id, connection_id, permission)
    SELECT entity_id, connection_id, 'READ'
    FROM guacamole_entity WHERE name = 'guacadmin';
END $$;
%{ endfor }
CONNECTIONS

# Create student user SQL
cat << 'STUDENT' > /opt/guacamole/init/004-create-student-user.sql
INSERT INTO guacamole_entity (name, type) VALUES ('student', 'USER');
INSERT INTO guacamole_user (entity_id, password_hash, password_salt, password_date)
SELECT entity_id, decode('4aa17c41c55b6b89ec4b62978d946e1b4210c03c3631343530313233', 'hex'),
       decode('1234567890', 'hex'), CURRENT_TIMESTAMP
FROM guacamole_entity WHERE name = 'student';
DO $$
DECLARE
    student_entity_id integer;
    connection_group_id integer;
BEGIN
    SELECT entity_id INTO student_entity_id FROM guacamole_entity WHERE name = 'student';
    SELECT connection_group_id INTO connection_group_id FROM guacamole_connection_group WHERE connection_group_name = 'Student VMs';
    INSERT INTO guacamole_connection_group_permission (entity_id, connection_group_id, permission)
    VALUES (student_entity_id, connection_group_id, 'READ');
    INSERT INTO guacamole_connection_permission (entity_id, connection_id, permission)
    SELECT student_entity_id, connection_id, 'READ' FROM guacamole_connection;
END $$;
STUDENT

# Setup permissions
usermod -aG docker ${var.vm_admin_username}
chown -R ${var.vm_admin_username}:${var.vm_admin_username} /opt/guacamole
chmod -R 755 /opt/guacamole

# Create Docker Compose file
cat << 'DOCKER' > docker-compose.yml
version: '3'
services:
  postgres:
    image: postgres:13
    container_name: postgres
    environment:
      POSTGRES_USER: guacamole
      POSTGRES_PASSWORD: ${var.guacamole_db_password}
      POSTGRES_DB: guacamole_db
    volumes:
      - postgres:/var/lib/postgresql/data
      - ./init:/docker-entrypoint-initdb.d
    restart: always
  guacd:
    image: guacamole/guacd
    container_name: guacd
    depends_on:
      - postgres
    restart: always
  guacamole:
    image: guacamole/guacamole
    container_name: guacamole
    depends_on:
      - guacd
      - postgres
    environment:
      GUACD_HOSTNAME: guacd
      POSTGRESQL_HOSTNAME: postgres
      POSTGRESQL_DATABASE: guacamole_db
      POSTGRESQL_USER: guacamole
      POSTGRESQL_PASSWORD: ${var.guacamole_db_password}
      WEBAPP_CONTEXT: training
    ports:
      - 8080:8080
    restart: always
volumes:
  postgres:
DOCKER

# Start services
systemctl enable docker
systemctl start docker
sleep 10
chmod 666 /var/run/docker.sock
docker-compose down -v

# Pull images first
docker pull postgres:13
docker pull guacamole/guacd
docker pull guacamole/guacamole

# Start containers
cd /opt/guacamole
docker-compose up -d
sleep 30
touch /var/log/guacamole-setup.log
chmod 644 /var/log/guacamole-setup.log
echo "=== Setup Time: $(date) ===" > /var/log/guacamole-setup.log
docker ps >> /var/log/guacamole-setup.log
docker-compose logs >> /var/log/guacamole-setup.log
docker exec postgres psql -U guacamole -d guacamole_db -c "\\dt" >> /var/log/guacamole-setup.log 2>&1
netstat -tulpn | grep 8080 >> /var/log/guacamole-setup.log
curl -s http://localhost:8080/guacamole/ >> /var/log/guacamole-setup.log 2>&1 