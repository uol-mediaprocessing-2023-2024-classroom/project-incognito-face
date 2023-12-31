<template>
  <v-app>
    <v-main>
      <HomePage
        :selectedImage="selectedImage"
        :selectedFilter="selectedFilter"
        :currentGallery="currentGallery"
        :currentFilters="currentFilters"
        :currentAlgorithms="currentAlgorithms"
        :faceResult="faceResult"
        :autoDetectionMode="autoDetectionMode"
        @loadImages="loadImages"
        @loadFilters="loadFilters"
        @selectImage="selectImage"
        @selectFilter="selectFilter"
        @uploadImage="uploadImage"
        @applyFilter="applyFilter"
        @runFaceDetection="runFaceDetection"
        @resetGallery="resetGallery"
        @toggleAutoDetectionMode="toggleAutoDetectionMode"
        @change_view="change_view"
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
      selectedImage: null,
      selectedFilter: null,
      currentGallery: [],
      currentFilters: [],
      currentAlgorithms: [],
      faceResult: [],
      faceRecognitionResult: [],
      autoDetectionMode: false,
      allImgData: [],
      limit: 60,
      loadedAmount: 0,
      backendHost: "http://127.0.0.1:8000",
    };
  },

  mounted() {
    this.selectedImage = require("@/assets/placeholder.json");
  },

  methods: {
    async loadImages() {
      const response = await fetch(this.backendHost + "/get-images");
      this.currentGallery = await response.json();
    },

    async loadFilters() {
      const response = await fetch(this.backendHost + "/get-filters");
      this.currentFilters = await response.json();
    },

    async selectImage(image) {
      // Deepcopy to keep the gallery intact
      this.selectedImage = JSON.parse(JSON.stringify(image));

      const requestBody = {
        base64: image.base64,
        hash: image.hash,
      };
      const response = await fetch(this.backendHost + "/generate-keypoints", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });
      await response.json();
    },

    async toggleAutoDetectionMode() {
      this.autoDetectionMode = !this.autoDetectionMode;
    },

    async uploadImage(file) {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const base64Encoded = e.target.result;
        const requestBody = {
          name: file.name,
          timestamp: new Date().toISOString(),
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
        this.currentGallery.push(newImage);
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
      this.selectedImage.base64 = jsonResponse.base64;
      if (this.autoDetectionMode) {
        await this.runFaceDetection(this.selectedImage);
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
        hash: image.hash,
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

    async runFaceRecognition(image) {
      if (image == null) {
        return;
      }
      const loading = require("@/assets/loading.json");
      this.faceRecognitionResult = {
        base64: loading.base64
      };
      const requestBody = {
        base64: image.base64,
        hash: image.hash,
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

    change_view() {

    },

    resetGallery() {
      this.selectedImage = require("@/assets/placeholder.json");
      this.currentGallery = [];
      this.faceResult = [];
    },
  },
};
</script>
