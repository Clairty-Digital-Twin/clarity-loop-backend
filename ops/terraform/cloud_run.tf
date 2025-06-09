resource "google_cloud_run_v2_service" "clarity_service" {
  name     = "clarity-backend"
  location = var.region
  
  template {
    containers {
      image = "us-central1-docker.pkg.dev/${var.project_id}/clarity-loop-backend/clarity-backend:latest"
      ports {
        container_port = 8000
      }
    }
  }

  traffic {
    percent         = 100
    type            = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
} 