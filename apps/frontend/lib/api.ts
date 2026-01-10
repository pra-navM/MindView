const API_BASE_URL = "http://localhost:8000";

export interface UploadResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface StatusResponse {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  progress: number;
  mesh_url: string | null;
  metadata_url: string | null;
  error: string | null;
}

export interface RegionInfo {
  name: string;
  label: string;
  color: [number, number, number];
  opacity: number;
  defaultVisible: boolean;
}

export interface MeshMetadata {
  job_id: string;
  regions: RegionInfo[];
  has_tumor: boolean;
  atlas_registered: boolean;
  input_type?: "segmentation" | "intensity";
}

export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Upload failed");
  }

  return response.json();
}

export async function getStatus(jobId: string): Promise<StatusResponse> {
  const response = await fetch(`${API_BASE_URL}/api/status/${jobId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to get status");
  }

  return response.json();
}

export async function getMeshMetadata(jobId: string): Promise<MeshMetadata> {
  const response = await fetch(`${API_BASE_URL}/api/mesh/${jobId}/metadata`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to get metadata");
  }

  return response.json();
}

export function getMeshUrl(jobId: string): string {
  return `${API_BASE_URL}/api/mesh/${jobId}`;
}
