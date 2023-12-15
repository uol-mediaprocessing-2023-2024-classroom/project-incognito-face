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
        @loadImages="loadImages"
        @loadFilters="loadFilters"
        @loadAlgorithms="loadAlgorithms"
        @selectImage="selectImage"
        @selectFilter="selectFilter"
        @applyFilter="applyFilter"
        @runFaceDetection="runFaceDetection"
        @resetGallery="resetGallery"
      />
    </v-main>
  </v-app>
</template>

<script>
import HomePage from "./components/HomePage";

export default {
  name: "App",

  components: {
    HomePage,
  },

  data() {
    return {
      selectedImage: null,
      selectedFilter: null,
      currentGallery: [],
      currentFilters: [],
      currentAlgorithms: [],
      faceResult: [],
      allImgData: [],
      limit: 60,
      loadedAmount: 0,
      backendHost: "http://127.0.0.1:8000"
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

    async loadAlgorithms() {
      const response = await fetch(this.backendHost + "/get-algorithms");
      this.currentAlgorithms = await response.json();
    },

    async selectImage(image) {
      // Deepcopy to keep the gallery intact
      this.selectedImage = JSON.parse(JSON.stringify(image));

      const requestBody = {
        base64: image.base64,
        hash: image.hash
      };
      const response = await fetch("http://127.0.0.1:8000/generate-keypoints", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });
      await response.json();
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
        hash: image.hash
      };
      const response = await fetch(this.backendHost + "/apply-filter", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });
      const jsonResponse = await response.json();
      this.selectedImage.base64 = jsonResponse.base64;
    },

    async runFaceDetection(image) {
      if (image == null) {
        return;
      }
      const requestBody = {
        base64: image.base64,
        hash: image.hash
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

    resetGallery() {
      this.selectedImage = require("@/assets/placeholder.json");
      this.currentGallery = [];
      this.faceResult = [];
    },
  },
};
</script>
