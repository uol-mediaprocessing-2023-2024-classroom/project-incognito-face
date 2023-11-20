<template>
    <v-app>
        <v-main>
            <!-- Communication between child and parent components can be done using props and events. Props are attributes passed from a parent to a child and can be used within it.
            A child component can emit events, which the parent then may react to. Here "selectedImage" is a prop passed to HomePage. HomePage emits the "fetchImgs" event,
            which triggers the fetchImgs method in App.vue. In this demo this is technically not needed, but since it's a core element of Vue I decided to include it.-->
            <HomePage :selectedImage="selectedImage" :currentGallery="currentGallery" :faceResult="faceResult" @loadImages="loadImages" @updateSelected="updateSelected" @getBlur="getBlur" @getFace="getFace" @resetGallery="resetGallery" />
        </v-main>
    </v-app>
</template>

<script>
import HomePage from "./components/HomePage";
import placeholder from "./assets/placeholder.jpg";



export default {
    name: "App",

    components: {
        HomePage,
    },

    data() {
        return {
            selectedImage: {
                url: placeholder,
                timestamp: 0
            },
            currentGallery: [],
            faceResult: {
                "viola-jones": {
                    url: "",
                    percentage: ""
                },
                "hog-svn": {
                    url: "",
                    percentage: ""
                }
            },
            allImgData: [],
            limit: 60,
            loadedAmount: 0
        };
    },

    methods: {

        /* 
          This method fetches the first 60 images from a user's gallery. 
          It first retrieves all image IDs, then it fetches specific image data. 
        */
        async loadImages() {
            const response = await fetch("http://127.0.0.1:8000/get-images");
            const imageData = await response.json();
            this.currentGallery = [];
            this.loadedAmount = 0;
            for (const imageInfo of imageData) {
                if (this.loadedAmount >= this.limit) break;
                this.loadedAmount++;
                this.currentGallery.push({
                    name: imageInfo.name,
                    timestamp: imageInfo.timestamp,
                    url: `http://127.0.0.1:8000/${imageInfo.url}`,
                });
            }
        },

        /* 
          This method updates the currently selected image. 
          It fetches the high-resolution URL of the selected image and updates the selectedImage property. 
        */
        async updateSelected(imgName) {
            const response = await fetch(`http://127.0.0.1:8000/get-img-data/${imgName}`);
            const imageData = await response.json();
            const image = this.currentGallery.find((obj) => obj.name === imgName);
            this.selectedImage = {
                name: imageData.name,
                timestamp: imageData.timestamp,
                url: `http://127.0.0.1:8000/${imageData.url}`,
            };
        },

        /* This method retrieves a blurred version of the selected image from the backend. */
        async getBlur(imgName) {
            const localUrl = `http://127.0.0.1:8000/get-blur/${imgName}`;
            const response = await fetch(localUrl);
            const imageBlob = await response.blob();
            const blurImgUrl = URL.createObjectURL(imageBlob);
            this.selectedImage.url = blurImgUrl;
        },

        async getFace(imgName) {
            const localUrl = `http://127.0.0.1:8000/get-face-data/${imgName}`;
            const response = await fetch(localUrl);
            const jsonResponse = await response.json();
            Object.keys(this.faceResult).forEach((key) => {
                this.faceResult[key].url = `http://127.0.0.1:8000/${jsonResponse[key].url}`;
                this.faceResult[key].percentage = jsonResponse[key].percentage;
            });
        },

        /* This method resets the current gallery and selected image. */
        resetGallery() {
            this.selectedImage = {
                url: placeholder,
                id: "placeholder"
            };
            this.currentGallery = [];
            this.faceResult = [];
        },
    },
};
</script>
