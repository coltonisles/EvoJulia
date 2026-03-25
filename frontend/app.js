

let genBtn = document.getElementById('generateBtn')
let imgIn = document.getElementById('imgInput')
let targetOutput = document.getElementById('targetDisplay')
let fractalOutput = document.getElementById('fractalDisplay')

genBtn.addEventListener('click', async () => {
    let formData = new FormData();
    console.log('button clicked')
    if (imgIn.files.length > 0) {
        targetOutput.src = URL.createObjectURL(imgIn.files[0])
        formData.append('file', imgIn.files[0])
        try {
            let response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            let data = await response.json();
            console.log("FULL SERVER RESPONSE:", data);

        } catch (error) {
            console.error("Error:", error);
        }
    }else{
        alert('Please select an image')
    }

})