<template>
  <v-app>
    <v-main>
      <HomePage
        :selectedFaceDetection="selectedFaceDetection"
        :selectedFilter="selectedFilter"
        :originalImage="originalImage"
        :modifiedImage="modifiedImage"
        :currentFilters="currentFilters"
        :currentAlgorithms="currentAlgorithms"
        :faceResult="faceResult"
        :faceRecognitionResult = "faceRecognitionResult"
        :autoDetectionMode="autoDetectionMode"
        @changeView="changeView"
        @resetImage="resetImage"
        @loadFilters="loadFilters"
        @selectFilter="selectFilter"
        @uploadImage="uploadImage"
        @applyFilter="applyFilter"
        @toggleAutoDetectionMode="toggleAutoDetectionMode"
        @runFaceDetection="runFaceDetection"
        @runFaceRecognition="runFaceRecognition"
      />
    </v-main>
  </v-app>
</template>

<script>
import HomePage from "./components/HomePage";
import loading from "@/assets/loading.json";

export default {
  name: "App",

  components: {
    HomePage,
  },

  data() {
    return {
      selectedFaceDetection: true,
      selectedFilter: null,
      originalImage: null,
      modifiedImage: null,
      currentFilters: [],
      currentAlgorithms: [],
      faceResult: [],
      faceRecognitionResult: [], //TODO: Vlt. noch später entfernen?
      autoDetectionMode: false,
      limit: 60,
      loadedAmount: 0,
      backendHost: "http://127.0.0.1:8000",
    };
  },

  methods: {
    async changeView() {
      this.selectedFaceDetection = !this.selectedFaceDetection;
      this.originalImage = null;
      this.modifiedImage = null;
      this.faceResult = [];
      this.faceRecognitionResult = [];
      this.currentAlgorithms = null;
      this.selectedFilter = null;
    },

    async resetImage(isOriginal, selectedFaceDetection) {
      if (selectedFaceDetection) {
          if (isOriginal) {
            this.originalImage = null;
            this.modifiedImage = null;
          }
          else {
            this.modifiedImage = JSON.parse(JSON.stringify(this.originalImage))
          }
        }
        else if (isOriginal) {
          this.originalImage = null;
        }
        else {
          this.modifiedImage = null;
        }
    },

    async loadFilters() {
      const response = await fetch(this.backendHost + "/get-filters");
      this.currentFilters = await response.json();
    },

    async toggleAutoDetectionMode() {
      this.autoDetectionMode = !this.autoDetectionMode;
    },

    async uploadImage(file, isOriginal, selectedFaceDetection) {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const base64Encoded = e.target.result;

        const requestBody = {
          name: file.name,
          base64: base64Encoded.split(",")[1],
        };

        const response = await fetch(this.backendHost + "/convert-image", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          console.error("HTTP Error:", response.status, response.statusText);
          return;
        }

        const newImage = await response.json();
        if (selectedFaceDetection) {
          this.originalImage = newImage;
          this.modifiedImage = JSON.parse(JSON.stringify(newImage));
        }
        else if (isOriginal) {
          this.originalImage = newImage;
        }
        else {
          this.modifiedImage = newImage;
        }

        const requestBody2 = {
          base64: newImage.base64,
          hash: newImage.hash,
        };
        const response2 = await fetch(this.backendHost + "/generate-keypoints", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody2),
        });
        await response2.json();
      };
      reader.readAsDataURL(file);
    },

    async selectFilter(filter) {
      this.selectedFilter = filter;
    },

    async applyFilter(image) {
      if (image == null || this.selectedFilter == null) {
        return;
      }
      const requestBody = {
        filter: this.selectedFilter.name,
        base64: image.base64,
        hash: image.hash,
      };
      // this.selectedImage.base64 = require("@/assets/loading.json").base64; // TODO: currently garbage css
      const response = await fetch(this.backendHost + "/apply-filter", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });
      const jsonResponse = await response.json();
      this.modifiedImage.base64 = jsonResponse.base64;
      if (this.autoDetectionMode) {
        await this.runFaceDetection(this.modifiedImage);
      }
    },

    async loadAlgorithms() {
      const response = await fetch(this.backendHost + "/get-algorithms");
      this.currentAlgorithms = await response.json();
    },

    async runFaceDetection(image) {
      if (image == null) {
        return;
      }
      if (this.currentAlgorithms == null || this.currentAlgorithms.length <= 0) {
        await this.loadAlgorithms();
      }
      const loading = require("@/assets/loading.json");
      this.faceResult = this.currentAlgorithms.map((algorithm) => ({
        name: algorithm.name,
        base64: loading.base64,
      }));
      const requestBody = {
        base64: image.base64,
      };
      const response = await fetch(this.backendHost + "/run-face-detection", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });
      this.faceResult = await response.json();
    },

    async runFaceRecognition(orig_image, mod_image) {
      if (orig_image == null || mod_image == null) {
        return;
      }
      const loading = require("@/assets/loading.json");
      this.faceRecognitionResult = [{
        base64: loading.base64
      }];
      const requestBody = {
        orig_hash: orig_image.hash,
        orig_base64: orig_image.base64,
        mod_base64: mod_image.base64,
      };
      const response = await fetch(this.backendHost + "/run-face-recognition", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });
      this.faceRecognitionResult = await response.json();
    },
  },
};
</script>
