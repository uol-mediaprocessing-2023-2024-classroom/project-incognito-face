<template>
  <v-app>
    <v-main>
      <HomePage
        :selectedImage="selectedImage"
        :selectedFilter="selectedFilter"
        :currentGallery="currentGallery"
        :currentFilters="currentFilters"
        :faceResult="faceResult"
        @loadImages="loadImages"
        @loadFilters="loadFilters"
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
      faceResult: {
        "viola-jones": {
          url: "",
          percentage: "",
        },
        "hog-svn": {
          url: "",
          percentage: "",
        },
      },
      allImgData: [],
      limit: 60,
      loadedAmount: 0,
    };
  },

  mounted() {
    this.selectedImage = require("@/assets/placeholder.json");
  },

  methods: {
    async loadImages() {
      const response = await fetch("http://127.0.0.1:8000/get-images");
      this.currentGallery = await response.json();
    },

    async loadFilters() {
      const response = await fetch("http://127.0.0.1:8000/get-filters");
      this.currentFilters = await response.json();
    },

    async loadAlgorithms() {
      const response = await fetch("http://127.0.0.1:8000/get-algorithms");
      this.currentFilters = await response.json();
    },

    async selectImage(image) {
      // Deepcopy to keep the gallery intact
      this.selectedImage = JSON.parse(JSON.stringify(image));
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
      };
      const response = await fetch("http://127.0.0.1:8000/apply-filter", {
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
      const localUrl = `http://127.0.0.1:8000/get-face-data/${imgName}`;
      const response = await fetch(localUrl);
      const jsonResponse = await response.json();
      Object.keys(this.faceResult).forEach((key) => {
        this.faceResult[key].url = `http://127.0.0.1:8000/${jsonResponse[key].url}`;
        this.faceResult[key].percentage = jsonResponse[key].percentage;
      });
    },

    resetGallery() {
      this.selectedImage = {
        url: placeholder,
        id: "placeholder",
      };
      this.currentGallery = [];
      this.faceResult = [];
    },
  },
};
</script>
