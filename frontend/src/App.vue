<template>
  <v-app>
    <v-main>
      <HomePage
        :selectedFaceDetection="selectedFaceDetection"
        :selectedFilter="selectedFilter"
        :originalImage="originalImage"
        :modifiedImage="modifiedImage"
        :modifiedImageButtonsHidden="modifiedImageButtonsHidden"
        :originalImageOutputFR="originalImageOutputFR"
        :modifiedImageOutputFR="modifiedImageOutputFR"
        :resultFR="resultFR"
        :currentFilters="currentFilters"
        :currentAlgorithms="currentAlgorithms"
        :faceResult="faceResult"
        :autoDetectionMode="autoDetectionMode"
        :faceOnly="faceOnly"
        @changeView="changeView"
        @resetImage="resetImage"
        @deleteImage="deleteImage"
        @loadFilters="loadFilters"
        @selectFilter="selectFilter"
        @uploadImage="uploadImage"
        @applyFilter="applyFilter"
        @toggleAutoDetectionMode="toggleAutoDetectionMode"
        @runFaceDetection="runFaceDetection"
        @runFaceRecognition="runFaceRecognition"
        @toggleFaceOnly="toggleFaceOnly"
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
      originalImageCopy: null,
      modifiedImage: null,
      modifiedImageCopy: null,
      modifiedImageButtonsHidden: false,
      originalImageOutputFR: null,
      modifiedImageOutputFR: null,
      resultFR: null,
      currentFilters: [],
      currentAlgorithms: [],
      faceResult: [],
      autoDetectionMode: false,
      faceOnly: false,
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
      this.originalImageOutputFR = null;
      this.modifiedImageOutputFR = null;
      this.faceResult = [];
      this.currentAlgorithms = null;
      this.selectedFilter = null;
    },

    async resetImage(isOriginal) {
      if (isOriginal) {
        this.originalImage = JSON.parse(JSON.stringify(this.originalImageCopy));
      }
      else {
        this.modifiedImage = JSON.parse(JSON.stringify(this.modifiedImageCopy));
      }
      this.currentAlgorithms = null;
      this.faceResult = [];
      this.originalImageOutputFR = null;
      this.modifiedImageOutputFR = null;
    },
    async deleteImage(isOriginal, selectedFaceDetection) {
      if (selectedFaceDetection) {
        this.originalImage = null;
        this.originalImageCopy = null;
        this.modifiedImage = null;
        this.modifiedImageCopy = null;
      }
      else if (isOriginal) {
        this.originalImage = null;
        this.originalImageCopy = null;
      }
      else {
        this.modifiedImage = null;
        this.modifiedImageCopy = null;
      }
      this.currentAlgorithms = null;
      this.faceResult = [];
      this.originalImageOutputFR = null;
      this.modifiedImageOutputFR = null;
    },

    async loadFilters() {
      const response = await fetch(this.backendHost + "/get-filters");
      this.currentFilters = await response.json();
    },

    async toggleAutoDetectionMode() {
      this.autoDetectionMode = !this.autoDetectionMode;
    },

    async toggleFaceOnly() {
      this.faceOnly = !this.faceOnly;
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
          this.originalImageCopy = JSON.parse(JSON.stringify(newImage));
          this.modifiedImageCopy = JSON.parse(JSON.stringify(newImage));
        }
        else if (isOriginal) {
          this.originalImage = newImage;
          this.originalImageCopy = JSON.parse(JSON.stringify(newImage));
        }
        else {
          this.modifiedImage = newImage;
          this.modifiedImageCopy = JSON.parse(JSON.stringify(newImage));
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
      const loading = require("@/assets/loading.json");
      this.modifiedImage = JSON.parse(JSON.stringify(image));
      this.modifiedImage.base64 = loading.base64;
      this.modifiedImageButtonsHidden = true;
      const requestBody = {
        filter: this.selectedFilter.name,
        base64: image.base64,
        hash: image.hash,
        face_only: this.faceOnly,
      };
      const response = await fetch(this.backendHost + "/apply-filter", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });
      const jsonResponse = await response.json();
      this.modifiedImage.base64 = jsonResponse.base64;
      this.modifiedImageButtonsHidden = false;
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
      this.originalImageOutputFR = JSON.parse(JSON.stringify(orig_image));
      this.modifiedImageOutputFR = JSON.parse(JSON.stringify(mod_image));
      this.originalImageOutputFR.base64 = loading.base64;
      this.modifiedImageOutputFR.base64 = loading.base64;
      this.resultFR = "";
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
      const jsonResponse = await response.json();

      this.originalImageOutputFR.base64 = jsonResponse.orig_base64;
      this.modifiedImageOutputFR.base64 = jsonResponse.mod_base64;
      this.resultFR = jsonResponse.metadata;
    },
  },
};
</script>
