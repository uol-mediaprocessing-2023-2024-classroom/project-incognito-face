<template>
    <v-app>
        <v-main>
            <!-- Communication between child and parent components can be done using props and events. Props are attributes passed from a parent to a child and can be used within it.
            A child component can emit events, which the parent then may react to. Here "selectedImage" is a prop passed to HomePage. HomePage emits the "fetchImgs" event,
            which triggers the fetchImgs method in App.vue. In this demo this is technically not needed, but since it's a core element of Vue I decided to include it.-->
            <HomePage :selectedImage="selectedImage" :currentGallery="currentGallery" @loadImages="loadImages" @updateSelected="updateSelected" @getBlur="getBlur" @getFace="getFace" @resetGallery="resetGallery" />
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

            // Reset current gallery and loaded amount before fetching new images
            this.currentGallery = [];
            this.loadedAmount = 0;

            // Fetch detailed image info for each image up to the limit
            for (const imageInfo of imageData) {
                if (this.loadedAmount >= this.limit) break;
                this.loadedAmount++;

                // Push image data to current gallery
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

            // Construct URL for fetching the high-resolution image
            const response = await fetch(`http://127.0.0.1:8000/get-img-data/${imgName}`);
            const imageData = await response.json();

            // Find the image data in the current gallery
            const image = this.currentGallery.find((obj) => obj.name === imgName);

            // Update the selected image with high-resolution URL and other details
            this.selectedImage = {
                name: imageData.name,
                timestamp: imageData.timestamp,
                url: `http://127.0.0.1:8000/${imageData.url}`,
            };
        },

        /* This method retrieves a blurred version of the selected image from the backend. */
        async getBlur(imgName) {
            console.log(imgName);
            const localUrl = `http://127.0.0.1:8000/get-blur/${imgName}`;

            // Fetch the blurred image
            const response = await fetch(localUrl);
            const imageBlob = await response.blob();
            const blurImgUrl = URL.createObjectURL(imageBlob);

            // Update the selected image with the URL of the blurred image
            this.selectedImage.url = blurImgUrl;
        },

        async getFace(imgName) {
            const localUrl = `http://127.0.0.1:8000/get-face/${imgName}`;

            // Fetch the face detected image
            const response = await fetch(localUrl);
            const imageBlob = await response.blob();
            const faceImgUrl = URL.createObjectURL(imageBlob);

            // Update the selected image with the URL of the face detected image
            this.selectedImage.url = faceImgUrl;
        },

        /* This method resets the current gallery and selected image. */
        resetGallery() {

            this.selectedImage = {
                url: placeholder,
                id: "placeholder"
            };
            this.currentGallery = [];
        },
    },
};
</script>
