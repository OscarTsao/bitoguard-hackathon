# BitoGuard AWS Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Internet                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Route 53 (DNS) │ (Optional)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   CloudFront    │ (Optional CDN)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  ALB (Public)   │
                    │  Port 80/443    │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        │         VPC (10.0.0.0/16)              │
        │                                         │
        │  ┌──────────────┐  ┌──────────────┐   │
        │  │Public Subnet │  │Public Subnet │   │
        │  │   AZ-1       │  │   AZ-2       │   │
        │  │              │  │              │   │
        │  │ NAT Gateway  │  │ NAT Gateway  │   │
        │  └──────┬───────┘  └──────┬───────┘   │
        │         │                  │            │
        │  ┌──────▼───────┐  ┌──────▼───────┐   │
        │  │Private Subnet│  │Private Subnet│   │
        │  │   AZ-1       │  │   AZ-2       │   │
        │  │              │  │              │   │
        │  │ ┌──────────┐ │  │ ┌──────────┐ │   │
        │  │ │ Backend  │ │  │ │ Backend  │ │   │
        │  │ │ ECS Task │ │  │ │ ECS Task │ │   │
        │  │ │ :8001    │ │  │ │ :8001    │ │   │
        │  │ └────┬─────┘ │  │ └────┬─────┘ │   │
        │  │      │       │  │      │       │   │
        │  │ ┌────▼─────┐ │  │ ┌────▼─────┐ │   │
        │  │ │Frontend  │ │  │ │Frontend  │ │   │
        │  │ │ ECS Task │ │  │ │ ECS Task │ │   │
        │  │ │ :3000    │ │  │ │ :3000    │ │   │
        │  │ └────┬─────┘ │  │ └────┬─────┘ │   │
        │  └──────┼───────┘  └──────┼───────┘   │
        │         │                  │            │
        │         └────────┬─────────┘            │
        │                  │                      │
        │         ┌────────▼────────┐             │
        │         │  EFS (Shared)   │             │
        │         │  - DuckDB       │             │
        │         │  - Artifacts    │             │
        │         └─────────────────┘             │
        └─────────────────────────────────────────┘
                         │
            ┌────────────┼────────────┐
            │            │            │
    ┌───────▼──────┐ ┌──▼──────┐ ┌──▼────────┐
    │  CloudWatch  │ │ Secrets │ │    ECR    │
    │    Logs      │ │ Manager │ │ (Images)  │
    └──────────────┘ └─────────┘ └───────────┘
```

## Components

### 1. Network Layer

**VPC (Virtual Private Cloud)**
- CIDR: 10.0.0.0/16
- 2 Availability Zones for high availability
- DNS hostnames enabled

**Public Subnets (2)**
- CIDR: 10.0.0.0/24, 10.0.1.0/24
- Internet Gateway attached
- Hosts: NAT Gateways, ALB

**Private Subnets (2)**
- CIDR: 10.0.10.0/24, 10.0.11.0/24
- NAT Gateway for outbound traffic
- Hosts: ECS tasks, EFS mount targets

**NAT Gateways (2)**
- One per AZ for high availability
- Enables private subnet internet access
- Elastic IPs attached

### 2. Compute Layer

**ECS Cluster**
- Fargate launch type (serverless)
- Container Insights enabled
- Capacity providers: FARGATE, FARGATE_SPOT

**Backend Service**
- Task Definition: 1 vCPU, 2GB RAM
- Desired Count: 2 (auto-scales 2-10)
- Port: 8001
- Health Check: /healthz
- EFS volume mounted at /mnt/efs

**Frontend Service**
- Task Definition: 0.5 vCPU, 1GB RAM
- Desired Count: 2 (auto-scales 2-10)
- Port: 3000
- Health Check: /

### 3. Load Balancing

**Application Load Balancer**
- Internet-facing
- HTTP (port 80) listener
- Path-based routing:
  - `/api/*` → Backend
  - `/*` → Frontend

**Target Groups**
- Backend TG: Port 8001, health check /healthz
- Frontend TG: Port 3000, health check /
- Deregistration delay: 30s

### 4. Storage Layer

**EFS (Elastic File System)**
- Encrypted at rest
- Lifecycle policy: IA after 30 days
- Mount targets in both AZs
- Access point for artifacts directory

**Purpose:**
- DuckDB database persistence
- Model artifacts storage
- Shared across all backend tasks

### 5. Container Registry

**ECR (Elastic Container Registry)**
- 2 repositories: backend, frontend
- Image scanning on push
- Lifecycle policy: Keep last 10 images
- Encryption: AES256

### 6. Security

**Security Groups**

1. ALB Security Group
   - Inbound: 80, 443 from 0.0.0.0/0
   - Outbound: All

2. Backend Security Group
   - Inbound: 8001 from ALB + Frontend SG
   - Outbound: All

3. Frontend Security Group
   - Inbound: 3000 from ALB SG
   - Outbound: All

4. EFS Security Group
   - Inbound: 2049 from Backend SG
   - Outbound: All

**IAM Roles**

1. Task Execution Role
   - ECR image pull
   - CloudWatch logs
   - Secrets Manager access

2. Backend Task Role
   - CloudWatch metrics
   - EFS mount

3. Frontend Task Role
   - CloudWatch logs

**Secrets Manager**
- API key for internal authentication
- Auto-generated 32-character key
- Referenced in task definitions

### 7. Monitoring

**CloudWatch Logs**
- Log groups: /ecs/bitoguard-prod-backend, /ecs/bitoguard-prod-frontend
- Retention: 7 days
- Log streams per task

**CloudWatch Metrics**
- ECS: CPU, Memory utilization
- ALB: Request count, latency, target health
- EFS: Throughput, IOPS

**CloudWatch Alarms**
- Backend CPU > 80%
- Backend Memory > 80%

### 8. Auto Scaling

**Application Auto Scaling**

Backend:
- Min: 2, Max: 10
- Target: 70% CPU, 70% Memory
- Scale-out cooldown: 60s
- Scale-in cooldown: 300s

Frontend:
- Min: 2, Max: 10
- Target: 70% CPU
- Scale-out cooldown: 60s
- Scale-in cooldown: 300s

## Data Flow

### User Request Flow

1. User → ALB (port 80)
2. ALB → Frontend Task (port 3000) or Backend Task (port 8001)
3. Frontend → Backend (via ALB DNS)
4. Backend → EFS (DuckDB queries)
5. Backend → External API (BitoPro)
6. Response back through chain

### Deployment Flow

1. Developer pushes code
2. GitHub Actions triggered
3. Docker images built
4. Images pushed to ECR
5. ECS service updated (force new deployment)
6. New tasks launched
7. Health checks pass
8. Old tasks drained and stopped

### Logging Flow

1. Container stdout/stderr
2. CloudWatch Logs agent
3. Log group/stream
4. Retention policy applied
5. Available for queries

## High Availability

**Multi-AZ Deployment**
- Resources spread across 2 AZs
- ALB distributes traffic
- EFS replicated across AZs
- NAT Gateway per AZ

**Fault Tolerance**
- If AZ-1 fails, AZ-2 continues
- Auto Scaling replaces failed tasks
- ALB health checks detect failures
- 30s deregistration delay for graceful shutdown

**Recovery Time**
- Task failure: ~60s (health check + new task)
- AZ failure: 0s (traffic shifts to healthy AZ)
- Region failure: Manual failover required

## Security Considerations

**Network Security**
- Private subnets for compute
- Security groups restrict traffic
- No direct internet access to tasks

**Data Security**
- EFS encryption at rest
- Secrets in Secrets Manager
- ECR image scanning
- VPC Flow Logs (optional)

**Access Control**
- IAM roles with least privilege
- API key authentication
- No hardcoded credentials

## Scalability

**Horizontal Scaling**
- Auto-scales 2-10 tasks per service
- Can increase max to 50+ if needed

**Vertical Scaling**
- Adjust CPU/memory in task definition
- Redeploy service

**Database Scaling**
- EFS scales automatically
- Consider RDS for larger datasets

## Cost Breakdown

See [COST_OPTIMIZATION.md](../../docs/COST_OPTIMIZATION.md) for detailed analysis.

**Monthly Costs (Base):**
- Compute (ECS): $90
- Load Balancer: $20
- NAT Gateway: $70
- Storage (EFS): $3
- Logs: $5
- Other: $10
- **Total: ~$198/month**

## Limitations

1. **DuckDB on EFS**: Not optimal for high concurrency
   - Consider RDS PostgreSQL for production
   - Or use S3 + Athena for analytics

2. **NAT Gateway Costs**: High for data-intensive workloads
   - Consider VPC endpoints for AWS services

3. **No CDN**: Frontend served directly from ALB
   - Add CloudFront for better performance

4. **Single Region**: No disaster recovery
   - Implement multi-region for critical workloads

## Future Enhancements

1. **HTTPS/SSL**
   - Add ACM certificate
   - Update ALB listener

2. **Custom Domain**
   - Route53 hosted zone
   - Alias record to ALB

3. **WAF**
   - AWS WAF for security
   - Rate limiting, IP filtering

4. **CloudFront**
   - CDN for frontend
   - Edge caching

5. **RDS**
   - Replace DuckDB with PostgreSQL
   - Better for production workloads

6. **ElastiCache**
   - Redis for caching
   - Session storage

7. **Multi-Region**
   - Active-passive setup
   - Route53 failover

8. **Backup**
   - AWS Backup for EFS
   - Automated snapshots

## Troubleshooting

**Tasks not starting**
- Check CloudWatch logs
- Verify ECR image exists
- Check IAM permissions
- Verify EFS mount targets

**Health checks failing**
- Check application logs
- Verify health endpoint responds
- Check security group rules
- Increase health check timeout

**High costs**
- Review NAT Gateway data transfer
- Check ECS task count
- Optimize log retention
- Use Fargate Spot

**Slow performance**
- Check ECS CPU/memory metrics
- Review ALB target response time
- Check EFS throughput
- Scale up tasks or resources
