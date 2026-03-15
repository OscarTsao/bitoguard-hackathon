# Step Functions State Machine for ML Pipeline

resource "aws_sfn_state_machine" "ml_pipeline" {
  name     = "${local.name_prefix}-ml-pipeline"
  role_arn = aws_iam_role.stepfunctions_execution.arn

  definition = jsonencode({
    Comment = "BitoGuard ML Pipeline Orchestration"
    StartAt = "ValidateConfiguration"
    States = {
      ValidateConfiguration = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.config_validator.arn
          Payload = {
            parameter_prefix = "/bitoguard/ml-pipeline"
          }
        }
        ResultPath = "$.validation"
        Next       = "CheckValidation"
        Retry = [
          {
            ErrorEquals     = ["Lambda.ServiceException", "Lambda.TooManyRequestsException"]
            IntervalSeconds = 2
            MaxAttempts     = 3
            BackoffRate     = 2.0
          }
        ]
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "NotifyFailure"
            ResultPath  = "$.error"
          }
        ]
      }

      CheckValidation = {
        Type = "Choice"
        Choices = [
          {
            Variable      = "$.validation.Payload.body"
            StringMatches = "*\"valid\":true*"
            Next          = "DataSyncStage"
          }
        ]
        Default = "NotifyFailure"
      }

      DataSyncStage = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          LaunchType     = "FARGATE"
          Cluster        = aws_ecs_cluster.main.arn
          TaskDefinition = aws_ecs_task_definition.ml_sync.arn
          NetworkConfiguration = {
            AwsvpcConfiguration = {
              Subnets        = aws_subnet.private[*].id
              SecurityGroups = [aws_security_group.ecs_tasks.id]
            }
          }
          CapacityProviderStrategy = [
            {
              CapacityProvider = "FARGATE_SPOT"
              Weight           = 70
            },
            {
              CapacityProvider = "FARGATE"
              Weight           = 30
            }
          ]
        }
        ResultPath = "$.sync"
        Next       = "FeatureEngineeringStage"
        Retry = [
          {
            ErrorEquals     = ["States.TaskFailed"]
            IntervalSeconds = 60
            MaxAttempts     = 2
            BackoffRate     = 2.0
          }
        ]
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "NotifyFailure"
            ResultPath  = "$.error"
          }
        ]
      }

      FeatureEngineeringStage = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          LaunchType     = "FARGATE"
          Cluster        = aws_ecs_cluster.main.arn
          TaskDefinition = aws_ecs_task_definition.ml_features.arn
          NetworkConfiguration = {
            AwsvpcConfiguration = {
              Subnets        = aws_subnet.private[*].id
              SecurityGroups = [aws_security_group.ecs_tasks.id]
            }
          }
          CapacityProviderStrategy = [
            {
              CapacityProvider = "FARGATE_SPOT"
              Weight           = 70
            },
            {
              CapacityProvider = "FARGATE"
              Weight           = 30
            }
          ]
        }
        ResultPath = "$.features"
        Next       = "PreprocessingStage"
        Retry = [
          {
            ErrorEquals     = ["States.TaskFailed"]
            IntervalSeconds = 60
            MaxAttempts     = 2
            BackoffRate     = 2.0
          }
        ]
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "NotifyFailure"
            ResultPath  = "$.error"
          }
        ]
      }

      PreprocessingStage = {
        Type     = "Task"
        Resource = "arn:aws:states:::sagemaker:createProcessingJob.sync"
        Parameters = {
          "ProcessingJobName.$" = "States.Format('bitoguard-preprocessing-{}', $$.Execution.Name)"
          RoleArn           = aws_iam_role.sagemaker_execution.arn
          AppSpecification = {
            ImageUri = "${aws_ecr_repository.backend.repository_url}:processing"
          }
          ProcessingResources = {
            ClusterConfig = {
              InstanceType   = "ml.c5.4xlarge"
              InstanceCount  = 1
              VolumeSizeInGB = 50
            }
          }
          ProcessingInputs = [
            {
              InputName = "raw-data"
              S3Input = {
                S3Uri                 = "s3://${aws_s3_bucket.artifacts.id}/features/"
                LocalPath             = "/opt/ml/processing/input"
                S3DataType            = "S3Prefix"
                S3InputMode           = "File"
                S3DataDistributionType = "FullyReplicated"
              }
            }
          ]
          ProcessingOutputConfig = {
            Outputs = [
              {
                OutputName = "processed-features"
                S3Output = {
                  S3Uri        = "s3://${aws_s3_bucket.artifacts.id}/features/processed/"
                  LocalPath    = "/opt/ml/processing/output"
                  S3UploadMode = "EndOfJob"
                }
              },
              {
                OutputName = "data-quality-report"
                S3Output = {
                  S3Uri        = "s3://${aws_s3_bucket.artifacts.id}/quality-reports/"
                  LocalPath    = "/opt/ml/processing/reports"
                  S3UploadMode = "EndOfJob"
                }
              }
            ]
          }
          StoppingCondition = {
            MaxRuntimeInSeconds = 3600
          }
          Environment = {
            DATA_SOURCE           = "efs"
            FEATURE_STORE_BUCKET  = aws_s3_bucket.artifacts.id
            "SNAPSHOT_ID.$"       = "States.Format('{}', $$.Execution.Name)"
          }
        }
        ResultPath = "$.preprocessing"
        Next       = "CheckSkipTraining"
        Retry = [
          {
            ErrorEquals     = ["States.TaskFailed"]
            IntervalSeconds = 60
            MaxAttempts     = 2
            BackoffRate     = 2.0
          }
        ]
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "NotifyFailure"
            ResultPath  = "$.error"
          }
        ]
      }

      CheckSkipTraining = {
        Type = "Choice"
        Choices = [
          {
            Variable      = "$.skip_training"
            BooleanEquals = true
            Next          = "ScoringStage"
          }
        ]
        Default = "CheckTuningEnabled"
      }

      CheckTuningEnabled = {
        Type = "Choice"
        Choices = [
          {
            Variable      = "$.enable_tuning"
            BooleanEquals = true
            Next          = "HyperparameterTuning"
          }
        ]
        Default = "ParallelTraining"
      }

      HyperparameterTuning = {
        Type = "Parallel"
        Branches = [
          {
            StartAt = "TuneLightGBM"
            States = {
              TuneLightGBM = {
                Type     = "Task"
                Resource = "arn:aws:states:::sagemaker:createHyperParameterTuningJob.sync"
                Parameters = {
                  "HyperParameterTuningJobName.$" = "States.Format('bitoguard-lgbm-tuning-{}', $$.Execution.Name)"
                  HyperParameterTuningJobConfig = {
                    Strategy = "Bayesian"
                    HyperParameterTuningJobObjective = {
                      Type       = "Maximize"
                      MetricName = "precision_at_100"
                    }
                    ResourceLimits = {
                      MaxNumberOfTrainingJobs  = 20
                      MaxParallelTrainingJobs  = 3
                    }
                    ParameterRanges = {
                      ContinuousParameterRanges = [
                        {
                          Name        = "learning_rate"
                          MinValue    = "0.01"
                          MaxValue    = "0.3"
                          ScalingType = "Logarithmic"
                        },
                        {
                          Name        = "subsample"
                          MinValue    = "0.6"
                          MaxValue    = "1.0"
                          ScalingType = "Linear"
                        },
                        {
                          Name        = "colsample_bytree"
                          MinValue    = "0.6"
                          MaxValue    = "1.0"
                          ScalingType = "Linear"
                        },
                        {
                          Name        = "reg_alpha"
                          MinValue    = "0.0"
                          MaxValue    = "1.0"
                          ScalingType = "Linear"
                        },
                        {
                          Name        = "reg_lambda"
                          MinValue    = "0.0"
                          MaxValue    = "1.0"
                          ScalingType = "Linear"
                        }
                      ]
                      IntegerParameterRanges = [
                        {
                          Name        = "num_leaves"
                          MinValue    = "20"
                          MaxValue    = "100"
                          ScalingType = "Linear"
                        },
                        {
                          Name        = "n_estimators"
                          MinValue    = "100"
                          MaxValue    = "500"
                          ScalingType = "Linear"
                        },
                        {
                          Name        = "min_data_in_leaf"
                          MinValue    = "10"
                          MaxValue    = "100"
                          ScalingType = "Linear"
                        },
                        {
                          Name        = "max_depth"
                          MinValue    = "3"
                          MaxValue    = "12"
                          ScalingType = "Linear"
                        }
                      ]
                    }
                  }
                  TrainingJobDefinition = {
                    StaticHyperParameters = {
                      model_type = "lgbm"
                    }
                    AlgorithmSpecification = {
                      TrainingImage     = "${aws_ecr_repository.backend.repository_url}:training"
                      TrainingInputMode = "File"
                      MetricDefinitions = [
                        {
                          Name  = "precision_at_100"
                          Regex = "precision_at_100: ([0-9\\\\.]+)"
                        },
                        {
                          Name  = "valid_logloss"
                          Regex = "valid_logloss: ([0-9\\\\.]+)"
                        },
                        {
                          Name  = "auc"
                          Regex = "auc: ([0-9\\\\.]+)"
                        }
                      ]
                    }
                    RoleArn = aws_iam_role.sagemaker_execution.arn
                    InputDataConfig = [
                      {
                        ChannelName = "training"
                        DataSource = {
                          S3DataSource = {
                            S3DataType             = "S3Prefix"
                            S3Uri                  = "s3://${aws_s3_bucket.artifacts.id}/features/processed/"
                            S3DataDistributionType = "FullyReplicated"
                          }
                        }
                        ContentType = "application/x-parquet"
                      }
                    ]
                    OutputDataConfig = {
                      S3OutputPath = "s3://${aws_s3_bucket.artifacts.id}/tuning-results/"
                    }
                    ResourceConfig = {
                      InstanceType   = "ml.c5.9xlarge"
                      InstanceCount  = 1
                      VolumeSizeInGB = 50
                    }
                    StoppingCondition = {
                      MaxRuntimeInSeconds = 3600
                    }
                    EnableManagedSpotTraining = false
                  }
                }
                ResultPath = "$.tuning.lgbm"
                End        = true
              }
            }
          },
          {
            StartAt = "TuneCatBoost"
            States = {
              TuneCatBoost = {
                Type     = "Task"
                Resource = "arn:aws:states:::sagemaker:createHyperParameterTuningJob.sync"
                Parameters = {
                  "HyperParameterTuningJobName.$" = "States.Format('bitoguard-catboost-tuning-{}', $$.Execution.Name)"
                  HyperParameterTuningJobConfig = {
                    Strategy = "Bayesian"
                    HyperParameterTuningJobObjective = {
                      Type       = "Maximize"
                      MetricName = "precision_at_100"
                    }
                    ResourceLimits = {
                      MaxNumberOfTrainingJobs  = 20
                      MaxParallelTrainingJobs  = 3
                    }
                    ParameterRanges = {
                      ContinuousParameterRanges = [
                        {
                          Name        = "learning_rate"
                          MinValue    = "0.01"
                          MaxValue    = "0.3"
                          ScalingType = "Logarithmic"
                        },
                        {
                          Name        = "subsample"
                          MinValue    = "0.6"
                          MaxValue    = "1.0"
                          ScalingType = "Linear"
                        },
                        {
                          Name        = "colsample_bytree"
                          MinValue    = "0.6"
                          MaxValue    = "1.0"
                          ScalingType = "Linear"
                        },
                        {
                          Name        = "l2_leaf_reg"
                          MinValue    = "1.0"
                          MaxValue    = "10.0"
                          ScalingType = "Linear"
                        }
                      ]
                      IntegerParameterRanges = [
                        {
                          Name        = "depth"
                          MinValue    = "4"
                          MaxValue    = "10"
                          ScalingType = "Linear"
                        },
                        {
                          Name        = "n_estimators"
                          MinValue    = "100"
                          MaxValue    = "500"
                          ScalingType = "Linear"
                        }
                      ]
                    }
                  }
                  TrainingJobDefinition = {
                    StaticHyperParameters = {
                      model_type = "catboost"
                    }
                    AlgorithmSpecification = {
                      TrainingImage     = "${aws_ecr_repository.backend.repository_url}:training"
                      TrainingInputMode = "File"
                      MetricDefinitions = [
                        {
                          Name  = "precision_at_100"
                          Regex = "precision_at_100: ([0-9\\\\.]+)"
                        },
                        {
                          Name  = "valid_logloss"
                          Regex = "valid_logloss: ([0-9\\\\.]+)"
                        },
                        {
                          Name  = "auc"
                          Regex = "auc: ([0-9\\\\.]+)"
                        }
                      ]
                    }
                    RoleArn = aws_iam_role.sagemaker_execution.arn
                    InputDataConfig = [
                      {
                        ChannelName = "training"
                        DataSource = {
                          S3DataSource = {
                            S3DataType             = "S3Prefix"
                            S3Uri                  = "s3://${aws_s3_bucket.artifacts.id}/features/processed/"
                            S3DataDistributionType = "FullyReplicated"
                          }
                        }
                        ContentType = "application/x-parquet"
                      }
                    ]
                    OutputDataConfig = {
                      S3OutputPath = "s3://${aws_s3_bucket.artifacts.id}/tuning-results/"
                    }
                    ResourceConfig = {
                      InstanceType   = "ml.c5.9xlarge"
                      InstanceCount  = 1
                      VolumeSizeInGB = 50
                    }
                    StoppingCondition = {
                      MaxRuntimeInSeconds = 3600
                    }
                    EnableManagedSpotTraining = false
                  }
                }
                ResultPath = "$.tuning.catboost"
                End        = true
              }
            }
          }
        ]
        ResultPath = "$.tuning_results"
        Next       = "AnalyzeTuning"
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "NotifyFailure"
            ResultPath  = "$.error"
          }
        ]
      }

      AnalyzeTuning = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.tuning_analyzer.arn
          Payload = {
            "execution_id.$"  = "$$.Execution.Name"
            "tuning_results.$" = "$.tuning_results"
          }
        }
        ResultPath = "$.tuning_analysis"
        Next       = "TrainStacker"
        Retry = [
          {
            ErrorEquals     = ["Lambda.ServiceException"]
            IntervalSeconds = 5
            MaxAttempts     = 2
          }
        ]
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "NotifyFailure"
            ResultPath  = "$.error"
          }
        ]
      }

      ParallelTraining = {
        Type = "Parallel"
        Branches = [
          {
            StartAt = "TrainLightGBM"
            States = {
              TrainLightGBM = {
                Type     = "Task"
                Resource = "arn:aws:states:::sagemaker:createTrainingJob.sync"
                Parameters = {
                  "TrainingJobName.$" = "States.Format('bitoguard-lgbm-{}', $$.Execution.Name)"
                  RoleArn         = aws_iam_role.sagemaker_execution.arn
                  AlgorithmSpecification = {
                    TrainingImage     = "${aws_ecr_repository.backend.repository_url}:training"
                    TrainingInputMode = "File"
                  }
                  InputDataConfig = [
                    {
                      ChannelName = "training"
                      DataSource = {
                        S3DataSource = {
                          S3DataType = "S3Prefix"
                          S3Uri      = "s3://${aws_s3_bucket.artifacts.id}/features/processed/"
                          S3DataDistributionType = "FullyReplicated"
                        }
                      }
                    }
                  ]
                  OutputDataConfig = {
                    S3OutputPath = "s3://${aws_s3_bucket.artifacts.id}/models/"
                  }
                  ResourceConfig = {
                    InstanceType   = "ml.c5.9xlarge"
                    InstanceCount  = 1
                    VolumeSizeInGB = 50
                  }
                  StoppingCondition = {
                    MaxRuntimeInSeconds = 3600
                  }
                  EnableManagedSpotTraining = false
                  HyperParameters = {
                    model_type = "lgbm"
                  }
                }
                End = true
              }
            }
          },
          {
            StartAt = "TrainCatBoost"
            States = {
              TrainCatBoost = {
                Type     = "Task"
                Resource = "arn:aws:states:::sagemaker:createTrainingJob.sync"
                Parameters = {
                  "TrainingJobName.$" = "States.Format('bitoguard-catboost-{}', $$.Execution.Name)"
                  RoleArn         = aws_iam_role.sagemaker_execution.arn
                  AlgorithmSpecification = {
                    TrainingImage     = "${aws_ecr_repository.backend.repository_url}:training"
                    TrainingInputMode = "File"
                  }
                  InputDataConfig = [
                    {
                      ChannelName = "training"
                      DataSource = {
                        S3DataSource = {
                          S3DataType = "S3Prefix"
                          S3Uri      = "s3://${aws_s3_bucket.artifacts.id}/features/processed/"
                          S3DataDistributionType = "FullyReplicated"
                        }
                      }
                    }
                  ]
                  OutputDataConfig = {
                    S3OutputPath = "s3://${aws_s3_bucket.artifacts.id}/models/"
                  }
                  ResourceConfig = {
                    InstanceType   = "ml.c5.9xlarge"
                    InstanceCount  = 1
                    VolumeSizeInGB = 50
                  }
                  StoppingCondition = {
                    MaxRuntimeInSeconds = 3600
                  }
                  EnableManagedSpotTraining = false
                  HyperParameters = {
                    model_type = "catboost"
                  }
                }
                End = true
              }
            }
          },
          {
            StartAt = "TrainIsolationForest"
            States = {
              TrainIsolationForest = {
                Type     = "Task"
                Resource = "arn:aws:states:::sagemaker:createTrainingJob.sync"
                Parameters = {
                  "TrainingJobName.$" = "States.Format('bitoguard-iforest-{}', $$.Execution.Name)"
                  RoleArn         = aws_iam_role.sagemaker_execution.arn
                  AlgorithmSpecification = {
                    TrainingImage     = "${aws_ecr_repository.backend.repository_url}:training"
                    TrainingInputMode = "File"
                  }
                  InputDataConfig = [
                    {
                      ChannelName = "training"
                      DataSource = {
                        S3DataSource = {
                          S3DataType = "S3Prefix"
                          S3Uri      = "s3://${aws_s3_bucket.artifacts.id}/features/processed/"
                          S3DataDistributionType = "FullyReplicated"
                        }
                      }
                    }
                  ]
                  OutputDataConfig = {
                    S3OutputPath = "s3://${aws_s3_bucket.artifacts.id}/models/"
                  }
                  ResourceConfig = {
                    InstanceType   = "ml.c5.4xlarge"
                    InstanceCount  = 1
                    VolumeSizeInGB = 50
                  }
                  StoppingCondition = {
                    MaxRuntimeInSeconds = 1800
                  }
                  EnableManagedSpotTraining = false
                  HyperParameters = {
                    model_type = "iforest"
                  }
                }
                End = true
              }
            }
          }
        ]
        ResultPath = "$.training"
        Next       = "TrainStacker"
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "NotifyFailure"
            ResultPath  = "$.error"
          }
        ]
      }

      TrainStacker = {
        Type     = "Task"
        Resource = "arn:aws:states:::sagemaker:createTrainingJob.sync"
        Parameters = {
          "TrainingJobName.$" = "States.Format('bitoguard-stacker-{}', $$.Execution.Name)"
          RoleArn         = aws_iam_role.sagemaker_execution.arn
          AlgorithmSpecification = {
            TrainingImage     = "${aws_ecr_repository.backend.repository_url}:training"
            TrainingInputMode = "File"
          }
          InputDataConfig = [
            {
              ChannelName = "training"
              DataSource = {
                S3DataSource = {
                  S3DataType = "S3Prefix"
                  S3Uri      = "s3://${aws_s3_bucket.artifacts.id}/features/processed/"
                  S3DataDistributionType = "FullyReplicated"
                }
              }
            }
          ]
          OutputDataConfig = {
            S3OutputPath = "s3://${aws_s3_bucket.artifacts.id}/models/"
          }
          ResourceConfig = {
            InstanceType   = "ml.c5.9xlarge"
            InstanceCount  = 1
            VolumeSizeInGB = 50
          }
          StoppingCondition = {
            MaxRuntimeInSeconds = 3600
          }
          EnableManagedSpotTraining = false
          HyperParameters = {
            model_type = "stacker"
            n_folds    = "5"
          }
        }
        ResultPath = "$.stacker_training"
        Next       = "RegisterModel"
        Retry = [
          {
            ErrorEquals     = ["States.TaskFailed"]
            IntervalSeconds = 60
            MaxAttempts     = 2
            BackoffRate     = 2.0
          }
        ]
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "NotifyFailure"
            ResultPath  = "$.error"
          }
        ]
      }

      RegisterModel = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.model_registry.arn
          Payload = {
            "execution_id.$"     = "$$.Execution.Name"
            "model_artifacts.$"  = "$.stacker_artifacts"
            model_type           = "stacker"
          }
        }
        ResultPath = "$.registration"
        Next       = "ScoringStage"
        Retry = [
          {
            ErrorEquals     = ["Lambda.ServiceException"]
            IntervalSeconds = 5
            MaxAttempts     = 2
          }
        ]
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "NotifyFailure"
            ResultPath  = "$.error"
          }
        ]
      }

      ScoringStage = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          LaunchType     = "FARGATE"
          Cluster        = aws_ecs_cluster.main.arn
          TaskDefinition = aws_ecs_task_definition.ml_scoring.arn
          NetworkConfiguration = {
            AwsvpcConfiguration = {
              Subnets        = aws_subnet.private[*].id
              SecurityGroups = [aws_security_group.ecs_tasks.id]
            }
          }
        }
        ResultPath = "$.scoring"
        Next       = "DriftDetection"
        Retry = [
          {
            ErrorEquals     = ["States.TaskFailed"]
            IntervalSeconds = 60
            MaxAttempts     = 2
            BackoffRate     = 2.0
          }
        ]
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "NotifyFailure"
            ResultPath  = "$.error"
          }
        ]
      }

      DriftDetection = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.drift_detector.arn
          Payload = {
            baseline_snapshot_id = "$.baseline_snapshot_id"
            current_snapshot_id  = "$.current_snapshot_id"
            bucket_name          = aws_s3_bucket.artifacts.id
          }
        }
        ResultPath = "$.drift"
        Next       = "PublishMetrics"
        Retry = [
          {
            ErrorEquals     = ["Lambda.ServiceException"]
            IntervalSeconds = 2
            MaxAttempts     = 3
            BackoffRate     = 2.0
          }
        ]
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "NotifyFailure"
            ResultPath  = "$.error"
          }
        ]
      }

      PublishMetrics = {
        Type     = "Task"
        Resource = "arn:aws:states:::aws-sdk:cloudwatch:putMetricData"
        Parameters = {
          Namespace = "BitoGuard/MLPipeline"
          MetricData = [
            {
              MetricName = "PipelineExecutionTime"
              Value      = "$.execution_time"
              Unit       = "Seconds"
            }
          ]
        }
        ResultPath = "$.metrics"
        Next       = "NotifySuccess"
      }

      NotifySuccess = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn = aws_sns_topic.ml_pipeline_notifications.arn
          Subject  = "BitoGuard ML Pipeline - Success"
          Message  = "ML Pipeline execution completed successfully"
        }
        End = true
      }

      NotifyFailure = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn = aws_sns_topic.critical_errors.arn
          Subject  = "BitoGuard ML Pipeline - Failure"
          Message  = "ML Pipeline execution failed. Check CloudWatch Logs for details."
        }
        Next = "FailState"
      }

      FailState = {
        Type  = "Fail"
        Error = "PipelineExecutionFailed"
        Cause = "ML Pipeline execution encountered an error"
      }
    }
  })

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.ml_pipeline.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-ml-pipeline"
  })
}

# Outputs
output "ml_pipeline_state_machine_arn" {
  description = "ARN of the ML pipeline state machine"
  value       = aws_sfn_state_machine.ml_pipeline.arn
}
