terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  credentials = file("gcp-credentials.json")
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  credentials = file("gcp-credentials.json")
}

resource "google_artifact_registry_repository" "clarity_repo" {
  provider      = google-beta
  location      = var.region
  repository_id = "clarity-loop-backend"
  description   = "Docker repository for CLARITY Loop Backend"
  format        = "DOCKER"
} 