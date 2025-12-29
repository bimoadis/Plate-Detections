<template>
  <div
    class="min-h-screen bg-gradient-to-b from-gray-100 to-green-100 font-sans"
  >
    <div class="max-w-6xl mx-auto py-12 px-6">
      <h1 class="text-4xl font-bold text-center mb-4">ANPR - Insight</h1>
      <p class="text-center text-gray-600 max-w-2xl mx-auto mb-10">
        ANPR-Insight adalah platform cerdas berbasis AI yang memungkinkan
        deteksi, pengenalan, dan analisis plat nomor kendaraan secara otomatis
        untuk kebutuhan pemantauan lalu lintas yang cepat dan efisien.
      </p>

      <div
        class="bg-white rounded-xl shadow-xl p-6 flex flex-col md:flex-row gap-6 transition-all"
      >
        <!-- Left Section -->
        <div class="flex-1 space-y-4">
          <div class="flex justify-between items-center">
            <h2 class="text-xl font-semibold">Smart Vehicle Monitoring</h2>
            <div v-if="hasVideo" class="flex gap-2">
              <button
                class="px-4 py-2 rounded-md text-red-600 border-red-600 border bg-red-50 hover:bg-red-600 hover:text-white transition"
                @click="resetVideo"
              >
                Reset Video
              </button>
            </div>
          </div>

          <!-- Upload Area -->
          <div
            v-if="!selectedFile"
            class="border-2 border-dashed border-emerald-500 rounded-lg p-6 text-center bg-emerald-50"
          >
            <div class="text-4xl mb-2">☁️</div>
            <p class="font-medium mb-1">Choose a file or drag & drop it here</p>
            <p class="text-sm text-gray-500 mb-4">
              Video formats (MP4, AVI, MOV), up to 50MB
            </p>
            <input
              ref="fileInput"
              type="file"
              accept="video/*"
              @change="handleFileSelect"
              class="hidden"
            />
            <button
              @click="triggerFileInput"
              class="px-5 py-2 rounded-full border border-emerald-600 text-emerald-600 hover:bg-emerald-600 hover:text-white transition"
            >
              Pilih Video
            </button>
          </div>

          <!-- Upload Progress -->
          <div v-if="selectedFile && !uploadComplete" class="space-y-4">
            <div
              v-if="uploading"
              class="border-2 border-emerald-500 rounded-lg p-4 bg-emerald-50"
            >
              <div class="flex items-center justify-between mb-3">
                <div>
                  <h3 class="text-sm font-semibold text-emerald-700">
                    📤 Mengupload Video...
                  </h3>
                  <p class="text-xs text-gray-600 mt-1">
                    File: {{ selectedFile.name }} ({{
                      formatFileSize(selectedFile.size)
                    }})
                  </p>
                </div>
                <button
                  @click="clearSelectedFile"
                  class="text-xs text-red-600 hover:text-red-700"
                  :disabled="uploading || processingDetection"
                >
                  ✕ Hapus
                </button>
              </div>
              <progress
                :value="progress"
                max="100"
                class="w-full h-2"
              ></progress>
              <p class="text-xs text-gray-600 mt-2 text-center">
                {{ progress }}%
              </p>
            </div>
          </div>

          <!-- After Upload Complete -->
          <div v-if="uploadComplete" class="space-y-4">
            <div
              class="border-2 border-emerald-500 rounded-lg p-4 bg-emerald-50"
            >
              <div class="flex items-center gap-2 mb-3">
                <span class="text-2xl">✅</span>
                <h3 class="text-lg font-semibold text-emerald-700">
                  Video Berhasil Diupload!
                </h3>
              </div>
              <video
                v-if="videoPreviewUrl"
                :src="videoPreviewUrl"
                controls
                class="w-full rounded-lg max-h-[300px] bg-black"
              ></video>
              <p class="text-xs text-gray-600 mt-2">
                File: {{ selectedFile?.name || uploadedFilename || "Video" }}
              </p>
            </div>

            <!-- Detection Status (Auto start setelah upload) -->
            <div v-if="processingDetection" class="text-center">
              <div class="inline-flex items-center gap-2 text-blue-600">
                <svg
                  class="animate-spin h-5 w-5"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    class="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    stroke-width="4"
                  ></circle>
                  <path
                    class="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
              </div>
              <p class="text-xs text-gray-500 mt-2">
                Proses ini mungkin memakan waktu beberapa menit
              </p>
            </div>

            <!-- Download Button - Show only when detection is complete -->
            <button
              v-if="detectionComplete"
              @click="downloadResults"
              class="w-full px-5 py-3 rounded-lg text-white bg-emerald-600 hover:bg-emerald-700 transition font-semibold text-lg mt-3 disabled:opacity-50 disabled:cursor-not-allowed"
              :disabled="downloading"
            >
              {{ downloading ? "⏳ Downloading..." : "📥 Download Hasil PDF" }}
            </button>
          </div>

          <p v-if="message" class="text-sm text-emerald-700">{{ message }}</p>
          <p v-if="error" class="text-sm text-red-600">{{ error }}</p>
        </div>

        <!-- Right Section -->
        <div
          class="flex-1 bg-gray-50 border border-dashed rounded-lg flex items-center justify-center text-gray-400"
        >
          <template v-if="hasil.length">
            <div class="text-left px-4 py-2 w-full">
              <div class="flex items-center justify-between mb-3">
                <h2
                  class="text-lg font-semibold text-gray-800 flex items-center gap-2"
                >
                  📊 Data Plat Nomor
                </h2>

                <!-- Filter Dropdown -->
                <div class="relative inline-block">
                  <select
                    v-model="filterType"
                    class="w-full px-4 py-2 pr-8 text-sm font-semibold text-emerald-700 bg-emerald-50 border-2 border-emerald-500 rounded-md cursor-pointer appearance-none transition-all duration-200 hover:bg-emerald-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:border-emerald-600"
                  >
                    <option value="all">Semua Plat</option>
                    <option value="putih">Plat Putih</option>
                    <option value="hitam">Plat Hitam</option>
                  </select>

                  <!-- Icon dropdown -->
                  <div
                    class="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-emerald-600"
                  >
                    <svg
                      class="w-4 h-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M19 9l-7 7-7-7"
                      />
                    </svg>
                  </div>
                </div>
              </div>

              <div class="max-h-[600px] overflow-y-auto space-y-5 pr-2">
                <div
                  v-for="(row, i) in filteredHasil"
                  :key="i"
                  class="bg-white rounded-xl shadow-md border border-gray-200 p-4 hover:shadow-lg transition-all"
                >
                  <!-- Header -->
                  <div class="flex justify-between items-center">
                    <h3 class="text-sm text-gray-700 font-semibold">
                      Waktu: {{ formatFrameToTime(row.Frame) }}
                    </h3>
                    <span
                      :class="[
                        'px-3 py-1 text-xs font-semibold rounded-full',
                        row['Label Plat']?.includes('putih')
                          ? 'bg-gray-200 text-gray-800'
                          : row['Label Plat']?.includes('hitam')
                          ? 'bg-gray-900 text-white'
                          : 'bg-blue-100 text-blue-800',
                      ]"
                    >
                      {{ row["Label Plat"] }}
                    </span>
                  </div>

                  <!-- Info -->
                  <div class="text-sm text-gray-600 space-y-1 mb-3">
                    <p v-if="row['SR Path'] || row['OCR SR']">
                      <span class="font-medium text-gray-800"
                        >Plat Terdeteksi :
                      </span>
                      <span
                        :class="[
                          row['OCR SR']
                            ? 'text-emerald-700 font-semibold'
                            : 'text-gray-400',
                        ]"
                      >
                        {{ row["OCR SR"] || "N/A" }}
                      </span>
                    </p>
                  </div>

                  <!-- Images Section -->
                  <div class="flex gap-3">
                    <!-- Deteksi - 55% width -->
                    <div
                      v-if="row['Deteksi Path']"
                      class="flex-shrink-0"
                      style="width: 60%"
                    >
                      <p
                        class="text-xs font-semibold text-blue-700 mb-1 text-center"
                      >
                        Deteksi
                      </p>
                      <div class="relative">
                        <img
                          :src="formatPath(row['Deteksi Path'])"
                          alt="Deteksi"
                          class="w-full object-contain max-h-[50vh] mx-auto rounded border-2 border-blue-300"
                          @error="handleImageError"
                        />
                      </div>
                    </div>

                    <!-- Crop dan SR - 45% width, vertikal (atas bawah) -->
                    <div
                      class="flex-shrink-0 flex flex-col gap-3"
                      style="width: 40%"
                    >
                      <!-- Crop -->
                      <!-- <div v-if="row['Crop Path']" class="text-center">
                      <p class="text-xs font-semibold text-gray-700 mb-1">
                        Crop
                      </p>
                      <div class="relative">
                        <img
                          :src="formatPath(row['Crop Path'])"
                          alt="Crop"
                            class="min-w-[60%] object-contain max-h-[14vh] mx-auto rounded border border-gray-200"
                          @error="handleImageError"
                        />
                        <p
                          v-if="row['OCR Crop']"
                          class="text-xs text-gray-600 mt-1 font-mono"
                        >
                          {{ row["OCR Crop"] }}
                        </p>
                      </div>
                    </div> -->

                      <!-- SR -->
                      <div v-if="row['SR Path']" class="text-center">
                        <p class="text-xs font-semibold text-emerald-700 mb-1">
                          Plat Nomor
                        </p>
                        <div class="relative">
                          <img
                            :src="formatPath(row['SR Path'])"
                            alt="SR"
                            class="max-w-[90%] object-contain max-h-[14vh] mx-auto rounded border-2 border-emerald-300"
                            @load="handleSRImageLoad(row)"
                            @error="handleImageError"
                          />
                          <p
                            v-if="row['OCR SR']"
                            class="text-xs text-emerald-700 mt-1 font-mono font-semibold"
                          >
                            {{ row["OCR SR"] }}
                          </p>
                        </div>
                        <!-- Button Detail Deteksi -->
                        <button
                          @click="openImageModal(row)"
                          class="mt-2 w-[90%] mx-auto px-3 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-md transition-colors"
                        >
                          Detail Deteksi
                        </button>
                      </div>

                      <!-- Threshold (if exists) -->
                      <div v-if="row['Threshold Path']" class="text-center">
                        <p class="text-xs font-semibold text-gray-700 mb-1">
                          Threshold
                        </p>
                        <img
                          :src="formatPath(row['Threshold Path'])"
                          alt="Threshold"
                          class="w-full object-contain max-h-[18vh] mx-auto rounded border border-gray-200"
                          @error="handleImageError"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </template>

          <template v-else>
            <div class="flex flex-col items-center">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                class="h-16 w-16 mb-2"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M9 17v-6a2 2 0 012-2h6m2 0a2 2 0 012 2v6a2 2 0 01-2 2H11a2 2 0 01-2-2z"
                />
              </svg>
              <p>Data Plat Nomor Kosong</p>
            </div>
          </template>
        </div>
      </div>
    </div>

    <!-- Image Modal Popup -->
    <div
      v-if="showImageModal"
      class="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-80 backdrop-blur-sm"
      @click.self="closeImageModal"
    >
      <div
        class="relative bg-white rounded-2xl shadow-2xl max-w-6xl max-h-[90vh] w-full mx-4 overflow-hidden flex flex-col"
      >
        <!-- Header -->
        <div
          class="bg-gradient-to-r from-blue-600 to-emerald-600 text-white px-6 py-4 flex justify-between items-center"
        >
          <div>
            <h3 class="text-xl font-bold">Detail Deteksi Plat Nomor</h3>
            <p class="text-sm opacity-90">
              Waktu:
              {{
                selectedRow?.Frame
                  ? formatFrameToTime(selectedRow.Frame)
                  : "N/A"
              }}
              | Label:
              {{ selectedRow?.["Label Plat"] || "N/A" }}
            </p>
          </div>
          <button
            @click="closeImageModal"
            class="text-white hover:text-gray-200 text-3xl font-bold bg-white bg-opacity-20 rounded-full w-10 h-10 flex items-center justify-center transition-all hover:bg-opacity-30"
            title="Close (ESC)"
          >
            ×
          </button>
        </div>

        <!-- Content -->
        <div class="flex-1 overflow-hidden flex gap-4 p-6">
          <!-- Left Side - 80% (Deteksi Image) -->
          <div class="flex-1 overflow-y-auto">
            <div
              v-if="selectedRow?.['Deteksi Path']"
              class="bg-gray-50 rounded-xl p-4 border-2 border-blue-300 h-full flex flex-col"
            >
              <!-- <div class="flex items-center gap-2 mb-3">
                <span class="text-2xl">🔍</span>
                <h4 class="text-lg font-semibold text-blue-700">
                  Gambar Deteksi
                </h4>
              </div> -->
              <div class="flex-1 flex items-center justify-center">
                <img
                  :src="formatPath(selectedRow['Deteksi Path'])"
                  alt="Deteksi"
                  class="max-w-full max-h-[75vh] object-contain rounded-lg shadow-lg"
                  @error="handleImageError"
                />
              </div>
            </div>
          </div>

          <!-- Right Sidebar - 20% (Crop, SR, Info) -->
          <div class="w-[20%] flex-shrink-0 overflow-y-auto space-y-4 pl-2">
            <!-- Crop Image -->
            <!-- <div
              v-if="selectedRow?.['Crop Path']"
              class="bg-gray-50 rounded-xl p-3 border-2 border-gray-300"
            >
              <div class="flex items-center gap-1 mb-1">
                <span class="text-lg">✂️</span>
                <h4 class="text-sm font-semibold text-gray-700">Crop</h4>
              </div>
              <img
                :src="formatPath(selectedRow['Crop Path'])"
                alt="Crop"
                class="w-full object-contain max-h-[20vh] mx-auto rounded-lg shadow-md mb-2"
                @error="handleImageError"
              />
              <div
                v-if="selectedRow?.['OCR Crop']"
                class="bg-white rounded-lg p-2 border border-gray-200"
              >
                <p class="text-xs text-gray-600 mb-1 font-semibold">
                  OCR: {{ selectedRow["OCR Crop"] }}
                </p>
              </div>
              <div
                v-else
                class="bg-gray-100 rounded-lg p-2 border border-gray-200"
              >
                <p class="text-xs text-gray-500 text-center italic">
                  OCR tidak tersedia
                </p>
              </div>
            </div> -->

            <!-- SR Image -->
            <div
              v-if="selectedRow?.['SR Path']"
              class="bg-gray-50 rounded-xl p-3 border-2 border-emerald-300"
            >
              <div class="flex items-center gap-1 mb-1">
                <h4 class="text-sm font-semibold text-emerald-700">
                  Plat Nomor Terdeteksi
                </h4>
              </div>
              <img
                :src="formatPath(selectedRow['SR Path'])"
                alt="SR"
                class="w-full object-contain max-h-[20vh] mx-auto rounded-lg shadow-md mb-2"
                @error="handleImageError"
              />
              <div
                v-if="selectedRow?.['OCR SR']"
                class="bg-emerald-50 rounded-lg p-2 border-2 border-emerald-300"
              >
                <p class="text-xs text-emerald-700 mb-1 font-semibold">
                  OCR : {{ selectedRow["OCR SR"] }}
                </p>
              </div>
              <div
                v-else
                class="bg-gray-100 rounded-lg p-2 border border-gray-200"
              >
                <p class="text-xs text-gray-500 text-center italic">
                  OCR tidak tersedia
                </p>
              </div>
            </div>

            <!-- Additional Info -->
            <div
              class="bg-gradient-to-r from-gray-50 to-gray-100 rounded-xl p-3 border border-gray-200"
            >
              <h4 class="text-xs font-semibold text-gray-700 mb-2">
                Informasi
              </h4>
              <div class="space-y-2 text-xs">
                <div>
                  <p class="text-gray-600">Confidence Plat:</p>
                  <p class="font-semibold text-gray-900">
                    {{ selectedRow?.["Conf YOLO"] || "N/A" }}
                  </p>
                </div>

                <div>
                  <p class="text-gray-600">Waktu:</p>
                  <p class="font-semibold text-gray-900">
                    {{
                      selectedRow?.Frame
                        ? formatFrameToTime(selectedRow.Frame)
                        : "N/A"
                    }}
                  </p>
                </div>
                <div>
                  <p class="text-gray-600">Label:</p>
                  <p class="font-semibold text-gray-900">
                    {{ selectedRow?.["Label Plat"] || "N/A" }}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from "vue";

const apiUrl = import.meta.env.VITE_API_URL;
const fileInput = ref(null);
const progress = ref(0);
const message = ref("");
const error = ref("");
const uploading = ref(false);
const uploadedFilename = ref("");
const hasil = ref([]);
const selectedFile = ref(null);
const videoPreviewUrl = ref("");
const uploadComplete = ref(false);
const processingDetection = ref(false);
const hasVideo = ref(false);
const filterType = ref("all");
const showImageModal = ref(false);
const selectedRow = ref(null);
const downloading = ref(false);

const triggerFileInput = () => fileInput.value.click();

const handleFileSelect = (e) => {
  const file = e.target.files[0];
  if (!file) {
    return;
  }

  if (!file.type.startsWith("video/")) {
    error.value = "Silakan pilih file video (MP4, AVI, MOV, dll).";
    return;
  }

  error.value = "";
  message.value = "";
  selectedFile.value = file;

  // Create preview URL
  if (videoPreviewUrl.value) {
    URL.revokeObjectURL(videoPreviewUrl.value);
  }
  videoPreviewUrl.value = URL.createObjectURL(file);
  uploadComplete.value = false;

  // Langsung upload dan deteksi setelah file dipilih
  startUploadAndDetect();
};

const clearSelectedFile = () => {
  selectedFile.value = null;
  if (videoPreviewUrl.value) {
    URL.revokeObjectURL(videoPreviewUrl.value);
    videoPreviewUrl.value = "";
  }
  uploadComplete.value = false;
  error.value = "";
  message.value = "";
  fileInput.value.value = "";
};

const formatFileSize = (bytes) => {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
};

const startUpload = () => {
  if (!selectedFile.value) {
    error.value = "Silakan pilih file terlebih dahulu.";
    return;
  }

  error.value = "";
  message.value = "";
  progress.value = 0;
  uploading.value = true;

  const formData = new FormData();
  formData.append("file", selectedFile.value);

  const xhr = new XMLHttpRequest();
  xhr.upload.addEventListener("progress", (event) => {
    if (event.lengthComputable) {
      progress.value = Math.round((event.loaded / event.total) * 100);
    }
  });
  xhr.onload = () => {
    uploading.value = false;
    if (xhr.status === 200) {
      const res = JSON.parse(xhr.responseText);
      message.value = res.message;
      uploadComplete.value = true;
      hasVideo.value = true;
    } else {
      error.value = "Upload gagal. Status: " + xhr.status;
    }
  };
  xhr.onerror = () => {
    uploading.value = false;
    error.value = "Terjadi kesalahan jaringan saat upload.";
  };
  xhr.open("POST", `${apiUrl}/upload-video`, true);
  xhr.send(formData);
};

// Fungsi untuk upload dan langsung deteksi
const startUploadAndDetect = () => {
  if (!selectedFile.value) {
    error.value = "Silakan pilih file terlebih dahulu.";
    return;
  }

  error.value = "";
  message.value = "📤 Mengupload video...";
  progress.value = 0;
  uploading.value = true;

  const formData = new FormData();
  formData.append("file", selectedFile.value);

  const xhr = new XMLHttpRequest();
  xhr.upload.addEventListener("progress", (event) => {
    if (event.lengthComputable) {
      progress.value = Math.round((event.loaded / event.total) * 100);
    }
  });
  xhr.onload = async () => {
    uploading.value = false;
    if (xhr.status === 200) {
      const res = JSON.parse(xhr.responseText);
      message.value = "✅ Upload selesai. Memulai deteksi...";
      uploadComplete.value = true;
      hasVideo.value = true;

      // Langsung mulai deteksi setelah upload selesai
      setTimeout(() => {
        startDetection();
      }, 500); // Delay kecil untuk memastikan upload benar-benar selesai
    } else {
      error.value = "Upload gagal. Status: " + xhr.status;
    }
  };
  xhr.onerror = () => {
    uploading.value = false;
    error.value = "Terjadi kesalahan jaringan saat upload.";
  };
  xhr.open("POST", `${apiUrl}/upload-video`, true);
  xhr.send(formData);
};

const startDetection = async () => {
  if (!uploadComplete.value) {
    error.value = "Silakan upload video terlebih dahulu.";
    return;
  }

  error.value = "";
  message.value = "";
  processingDetection.value = true;

  try {
    // Start detection
    const res = await fetch(`${apiUrl}/detect-plate`, {
      method: "POST",
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.error || `HTTP ${res.status}`);
    }

    const data = await res.json();
    message.value = data.message;

    // Poll status setiap 2 detik
    const statusInterval = setInterval(async () => {
      try {
        const statusRes = await fetch(`${apiUrl}/detect-status`);
        const statusData = await statusRes.json();

        if (statusData.error) {
          error.value = statusData.error;
          processingDetection.value = false;
          clearInterval(statusInterval);
          return;
        }

        if (!statusData.is_processing) {
          // Deteksi selesai
          processingDetection.value = false;
          clearInterval(statusInterval);
          message.value = statusData.message;
          hasVideo.value = true;

          // Refresh hasil
          setTimeout(() => {
            getHasil();
          }, 1000);
        } else {
          // Masih processing
          message.value = statusData.message;
        }
      } catch (err) {
        console.error("Error checking status:", err);
      }
    }, 2000);

    // Stop checking after 10 minutes
    setTimeout(() => {
      clearInterval(statusInterval);
      if (processingDetection.value) {
        processingDetection.value = false;
        message.value =
          "⏳ Proses deteksi masih berjalan. Silakan refresh hasil secara manual.";
      }
    }, 600000); // 10 menit
  } catch (err) {
    console.error("Error saat deteksi:", err);
    error.value =
      "Terjadi kesalahan saat deteksi: " + (err.message || err.toString());
    processingDetection.value = false;
  }
};

const getHasil = async () => {
  const res = await fetch(`${apiUrl}/get-results`);
  const data = await res.json();
  console.log("DATA HASIL:", data);
  hasil.value = data.data || [];

  // Update hasVideo berdasarkan hasil
  hasVideo.value = hasil.value.length > 0;
};

const handleSRImageLoad = (row) => {
  // Handle SR image load if needed
};

const handleImageError = (e) => {
  // Handle image load error
  e.target.style.display = "none";
};

const resetVideo = async () => {
  const res = await fetch(`${apiUrl}/clear-results`, { method: "DELETE" });
  const data = await res.json();
  alert(data.message);
  hasil.value = [];
  message.value = "";
  uploadedFilename.value = "";
  selectedFile.value = null;
  uploadComplete.value = false;
  processingDetection.value = false;
  hasVideo.value = false;
  if (videoPreviewUrl.value) {
    URL.revokeObjectURL(videoPreviewUrl.value);
    videoPreviewUrl.value = "";
  }
  fileInput.value.value = "";
};

const formatPath = (path) => {
  if (!path) return "";
  const cleanPath = path.replace(/\\/g, "/");
  // Hilangkan double slash jika ada
  return `${apiUrl}${cleanPath.startsWith("/") ? "" : "/"}${cleanPath}`;
};

const openImageModal = (row) => {
  selectedRow.value = row;
  showImageModal.value = true;
  // Prevent body scroll when modal is open
  document.body.style.overflow = "hidden";
};

const closeImageModal = () => {
  showImageModal.value = false;
  selectedRow.value = null;
  // Restore body scroll
  document.body.style.overflow = "auto";
};

// Konversi frame ke format waktu (MM:SS atau HH:MM:SS)
// Asumsi: 30 fps (frame rate standar)
const formatFrameToTime = (frameNumber) => {
  if (!frameNumber || isNaN(frameNumber)) return "00:00";

  const fps = 25; // Frame per second
  const totalSeconds = Math.floor(frameNumber / fps);

  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(
      2,
      "0"
    )}:${String(seconds).padStart(2, "0")}`;
  } else {
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(
      2,
      "0"
    )}`;
  }
};

// Group results by frame
const groupByFrame = (data) => {
  const grouped = {};
  data.forEach((row) => {
    const frame = row.Frame || "unknown";
    if (!grouped[frame]) {
      grouped[frame] = [];
    }
    grouped[frame].push(row);
  });
  return grouped;
};

// Load image from URL
const loadImage = (url) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
};

// Combine detection and SR images side by side
const combineImages = async (deteksiUrl, srUrl) => {
  try {
    const [deteksiImg, srImg] = await Promise.all([
      loadImage(deteksiUrl),
      srUrl ? loadImage(srUrl) : loadImage(deteksiUrl), // Fallback to deteksi if SR not available
    ]);

    // Calculate canvas dimensions
    const deteksiHeight = deteksiImg.height;
    const deteksiWidth = deteksiImg.width;
    const srHeight = srImg.height;
    const srWidth = srImg.width;

    // Use the maximum height for both images
    const maxHeight = Math.max(deteksiHeight, srHeight);
    const deteksiScale = maxHeight / deteksiHeight;
    const srScale = maxHeight / srHeight;

    const scaledDeteksiWidth = deteksiWidth * deteksiScale;
    const scaledSrWidth = srWidth * srScale;

    // Create canvas
    const canvas = document.createElement("canvas");
    canvas.width = scaledDeteksiWidth + scaledSrWidth;
    canvas.height = maxHeight;
    const ctx = canvas.getContext("2d");

    // Fill white background
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw detection image (left)
    ctx.drawImage(deteksiImg, 0, 0, scaledDeteksiWidth, maxHeight);

    // Draw SR image (right)
    ctx.drawImage(srImg, scaledDeteksiWidth, 0, scaledSrWidth, maxHeight);

    return canvas;
  } catch (error) {
    console.error("Error combining images:", error);
    return null;
  }
};

// Convert image to base64 using fetch API to handle CORS better
const imageToBase64 = async (url) => {
  try {
    // Try using fetch first (better CORS handling)
    const response = await fetch(url, {
      mode: "cors",
      credentials: "omit",
      cache: "no-cache",
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const blob = await response.blob();

    // Validate blob type
    if (!blob.type.startsWith("image/")) {
      throw new Error(`Invalid image type: ${blob.type}`);
    }

    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        if (reader.result) {
          resolve(reader.result);
        } else {
          reject(new Error("Failed to read image data"));
        }
      };
      reader.onerror = () => reject(new Error("FileReader error"));
      reader.readAsDataURL(blob);
    });
  } catch (fetchError) {
    // Fallback to Image method if fetch fails
    console.warn("Fetch method failed, trying Image method:", fetchError);
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";

      // Add timeout
      const timeout = setTimeout(() => {
        reject(new Error("Image load timeout"));
      }, 10000); // 10 seconds timeout

      img.onload = () => {
        clearTimeout(timeout);
        try {
          const canvas = document.createElement("canvas");
          canvas.width = img.width;
          canvas.height = img.height;
          const ctx = canvas.getContext("2d");

          // Fill white background for transparent images
          ctx.fillStyle = "#FFFFFF";
          ctx.fillRect(0, 0, canvas.width, canvas.height);

          ctx.drawImage(img, 0, 0);

          // Determine format based on file extension or use JPEG
          const format = url.toLowerCase().endsWith(".png")
            ? "image/png"
            : "image/jpeg";
          const quality = format === "image/png" ? undefined : 0.9;

          resolve(canvas.toDataURL(format, quality));
        } catch (canvasError) {
          clearTimeout(timeout);
          reject(new Error(`Canvas error: ${canvasError.message}`));
        }
      };
      img.onerror = (error) => {
        clearTimeout(timeout);
        reject(
          new Error(
            `Image load error: ${error.message || "Failed to load image"}`
          )
        );
      };
      img.src = url;
    });
  }
};

// Load jsPDF from CDN
const loadJsPDF = () => {
  return new Promise((resolve, reject) => {
    if (window.jspdf) {
      resolve(window.jspdf);
      return;
    }
    const script = document.createElement("script");
    script.src =
      "https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js";
    script.onload = () => {
      if (window.jspdf) {
        resolve(window.jspdf);
      } else {
        reject(new Error("jsPDF failed to load"));
      }
    };
    script.onerror = () => reject(new Error("Failed to load jsPDF script"));
    document.head.appendChild(script);
  });
};

// Download results as PDF grouped by frame
const downloadResults = async () => {
  if (hasil.value.length === 0) {
    alert("Tidak ada data untuk didownload");
    return;
  }

  downloading.value = true;

  try {
    // Load jsPDF
    const jsPDF = await loadJsPDF();
    const pdf = new jsPDF.jsPDF({
      orientation: "landscape",
      unit: "mm",
      format: "a4",
    });

    const grouped = groupByFrame(hasil.value);
    const frames = Object.keys(grouped).sort(
      (a, b) => parseInt(a) - parseInt(b)
    );

    if (frames.length === 0) {
      alert("Tidak ada frame yang valid untuk didownload");
      downloading.value = false;
      return;
    }

    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const margin = 10;
    const contentWidth = pageWidth - 2 * margin;
    const imageHeight = 60; // Height for each image row in mm
    const textHeight = 8; // Height for OCR text

    let yPosition = margin;
    let pageNumber = 1;
    let processedFrames = 0;
    let skippedFrames = 0;

    for (const frame of frames) {
      const frameData = grouped[frame];

      // Find first detection with deteksi path for this frame
      const firstDetection = frameData.find((row) => row["Deteksi Path"]);

      if (!firstDetection) {
        skippedFrames++;
        continue;
      }

      // Calculate needed height: deteksi image + all SR images (2x smaller) + OCR texts
      const srImageHeight = imageHeight / 2; // 2x smaller
      const srCount = frameData.filter((row) => row["SR Path"]).length;
      const neededHeight =
        imageHeight + // Detection image
        srCount * (srImageHeight + 4) + // All SR images with spacing
        frameData.length * (textHeight + 4) + // OCR texts for each detection
        15; // Frame header

      if (yPosition + neededHeight > pageHeight - margin) {
        pdf.addPage();
        yPosition = margin;
        pageNumber++;
      }

      // Frame header
      pdf.setFontSize(14);
      pdf.setFont("helvetica", "bold");
      pdf.text(
        `Waktu: ${formatFrameToTime(parseInt(frame))}`,
        margin,
        yPosition
      );
      yPosition += 8;

      try {
        // Load and add detection image (left side - 60% width) - only once per frame
        const deteksiUrl = formatPath(firstDetection["Deteksi Path"]);
        if (!deteksiUrl) {
          skippedFrames++;
          continue;
        }

        let deteksiBase64;
        try {
          deteksiBase64 = await imageToBase64(deteksiUrl);
          // Validate base64 string
          if (!deteksiBase64 || !deteksiBase64.startsWith("data:image")) {
            throw new Error("Invalid image data");
          }
        } catch (imgError) {
          console.error(
            `Error loading detection image for frame ${frame}:`,
            imgError
          );
          skippedFrames++;
          continue;
        }

        const deteksiWidth = contentWidth * 0.6; // 60% width untuk gambar deteksi
        const deteksiStartY = yPosition;

        // Determine image format from base64
        const imageFormat = deteksiBase64.includes("data:image/png")
          ? "PNG"
          : "JPEG";

        pdf.addImage(
          deteksiBase64,
          imageFormat,
          margin,
          deteksiStartY,
          deteksiWidth,
          imageHeight
        );

        // Add detection label
        pdf.setFontSize(8);
        pdf.setFont("helvetica", "normal");
        pdf.text("Deteksi", margin + 2, deteksiStartY - 2);

        // Right side: Add all SR images (2x smaller) vertically
        let srYPosition = deteksiStartY;
        const srWidth = contentWidth * 0.4; // 40% width untuk gambar plat
        const srStartX = margin + deteksiWidth;

        // Process all detections in this frame to get all SR images
        for (let i = 0; i < frameData.length; i++) {
          const row = frameData[i];
          const srPath = row["SR Path"];

          if (srPath) {
            try {
              const srUrl = formatPath(srPath);
              if (!srUrl) continue;

              const srBase64 = await imageToBase64(srUrl);
              // Validate base64 string
              if (!srBase64 || !srBase64.startsWith("data:image")) {
                console.warn(
                  `Invalid SR image data for frame ${frame}, detection ${i + 1}`
                );
                continue;
              }

              // Determine image format from base64
              const srImageFormat = srBase64.includes("data:image/png")
                ? "PNG"
                : "JPEG";

              // Add SR image (2x smaller)
              pdf.addImage(
                srBase64,
                srImageFormat,
                srStartX,
                srYPosition,
                srWidth,
                srImageHeight
              );

              // Add SR label with index
              pdf.setFontSize(7);
              pdf.text(`Plat Nomor ${i + 1}`, srStartX + 2, srYPosition - 1);

              srYPosition += srImageHeight + 4; // Space between SR images
            } catch (e) {
              console.error(
                `Error loading SR image ${i + 1} for frame ${frame}:`,
                e
              );
              // Continue with next SR image instead of breaking
            }
          }
        }

        // Add OCR information below images for all detections
        yPosition = deteksiStartY + imageHeight + 5;
        pdf.setFontSize(7);
        pdf.setFont("helvetica", "bold");

        for (let i = 0; i < frameData.length; i++) {
          const row = frameData[i];
          const ocrSR = row["OCR SR"] || "N/A";
          const conf = row["Conf YOLO"] || "N/A";
          const label = row["Label Plat"] || "N/A";

          // Detection number
          pdf.text(`Deteksi ${i + 1}:`, margin, yPosition);
          yPosition += 4;

          // OCR SR
          pdf.setFont("helvetica", "normal");
          pdf.text(`  Plat Nomor Terdeteksi: ${ocrSR}`, margin, yPosition);
          yPosition += 3;

          // Additional info
          pdf.setFont("helvetica", "bold");
          pdf.text(
            `  Confidence: ${conf} | Label: ${label}`,
            margin,
            yPosition
          );
          yPosition += 5; // Space before next detection info
        }

        processedFrames++;
      } catch (error) {
        console.error(`Error processing frame ${frame}:`, error);
        skippedFrames++;
      }

      // Add space between frames
      yPosition += 5;
    }

    // Check if any frames were processed
    if (processedFrames === 0) {
      alert(
        "❌ Tidak ada frame yang berhasil diproses. Pastikan gambar dapat diakses."
      );
      downloading.value = false;
      return;
    }

    // Save PDF
    const timestamp = new Date()
      .toISOString()
      .replace(/[:.]/g, "-")
      .slice(0, -5);

    try {
      pdf.save(`hasil_deteksi_${timestamp}.pdf`);
      const message =
        skippedFrames > 0
          ? `✅ Download selesai! PDF berisi ${processedFrames} frame(s) dengan total ${hasil.value.length} deteksi. (${skippedFrames} frame dilewati karena error)`
          : `✅ Download selesai! PDF berisi ${processedFrames} frame(s) dengan total ${hasil.value.length} deteksi.`;
      alert(message);
    } catch (saveError) {
      console.error("Error saving PDF:", saveError);
      // Try alternative method - create blob and download
      try {
        const pdfBlob = pdf.output("blob");
        const url = URL.createObjectURL(pdfBlob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `hasil_deteksi_${timestamp}.pdf`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        const message =
          skippedFrames > 0
            ? `✅ Download selesai! PDF berisi ${processedFrames} frame(s) dengan total ${hasil.value.length} deteksi. (${skippedFrames} frame dilewati karena error)`
            : `✅ Download selesai! PDF berisi ${processedFrames} frame(s) dengan total ${hasil.value.length} deteksi.`;
        alert(message);
      } catch (blobError) {
        console.error("Error creating blob:", blobError);
        alert(
          "❌ Error saat menyimpan PDF. Silakan coba lagi atau periksa console untuk detail."
        );
      }
    }
  } catch (error) {
    console.error("Error downloading results:", error);
    alert("❌ Error saat download: " + (error.message || error.toString()));
  } finally {
    downloading.value = false;
  }
};

onMounted(() => {
  getHasil();
  // Cek apakah sudah ada hasil (ada video yang sudah diupload)
  checkHasVideo();
  // Load current video jika ada
  loadCurrentVideo();

  // Add ESC key listener for modal
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && showImageModal.value) {
      closeImageModal();
    }
  });
});

const checkHasVideo = async () => {
  const res = await fetch(`${apiUrl}/get-results`);
  const data = await res.json();
  hasVideo.value = data.data && data.data.length > 0;
};

const loadCurrentVideo = async () => {
  try {
    const res = await fetch(`${apiUrl}/get-current-video`);
    const data = await res.json();

    if (data.exists && data.video_info) {
      // Set video info
      const originalName =
        data.video_info.original_filename || data.video_info.filename;
      uploadedFilename.value = originalName;
      uploadComplete.value = true;
      hasVideo.value = true;

      // Load video preview dari server
      if (data.video_url) {
        const videoUrl = `${apiUrl}${data.video_url}`;
        videoPreviewUrl.value = videoUrl;

        // Create a dummy File object untuk UI consistency
        // Ini diperlukan agar selectedFile?.name bisa digunakan di template
        const dummyFile = {
          name: originalName,
          size: 0, // Size tidak diketahui dari server
          type: "video/mp4", // Default type
        };
        selectedFile.value = dummyFile;
      }
    }
  } catch (error) {
    console.error("Error loading current video:", error);
    // Tidak perlu error handling, hanya log
  }
};

// Filter menggunakan v-model langsung pada select dropdown, tidak perlu toggle function

// Computed property untuk mengecek apakah deteksi sudah selesai
const detectionComplete = computed(() => {
  return !processingDetection.value && hasil.value.length > 0;
});

// Computed property untuk filter hasil
const filteredHasil = computed(() => {
  let filtered = [];

  if (filterType.value === "all") {
    filtered = hasil.value;
  } else if (filterType.value === "hitam") {
    filtered = hasil.value.filter((row) =>
      row["Label Plat"]?.toLowerCase().includes("hitam")
    );
  } else if (filterType.value === "putih") {
    filtered = hasil.value.filter((row) =>
      row["Label Plat"]?.toLowerCase().includes("putih")
    );
  } else {
    filtered = hasil.value;
  }

  // Urutkan berdasarkan panjang plat nomor terdeteksi (OCR SR) dari panjang ke pendek
  return filtered.sort((a, b) => {
    const ocrA = (a["OCR SR"] || "").toString().trim();
    const ocrB = (b["OCR SR"] || "").toString().trim();
    const lengthA = ocrA.length;
    const lengthB = ocrB.length;

    // Jika panjang sama, urutkan berdasarkan nilai string (alphabetical)
    if (lengthB === lengthA) {
      return ocrB.localeCompare(ocrA);
    }

    // Urutkan dari panjang ke pendek
    return lengthB - lengthA;
  });
});
</script>

<style scoped>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap");
:root {
  font-family: "Inter", sans-serif;
}
</style>
