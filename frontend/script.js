const backendURL = "http://127.0.0.1:8000"; // your local FastAPI server

const lettersDiv = document.getElementById("letters");
const wordsDiv = document.getElementById("words");
const sentenceP = document.getElementById("sentence");

let selectedLetter = "";
let selectedWord = "";
let selectedSentence = "";

window.onload = async () => {
  const res = await fetch(`${backendURL}/letters`);
  const data = await res.json();
  data.letters.forEach(letter => {
    const btn = document.createElement("button");
    btn.innerText = letter;
    btn.onclick = () => fetchWords(letter);
    lettersDiv.appendChild(btn);
  });
};

async function fetchWords(letter) {
  selectedLetter = letter;
  const res = await fetch(`${backendURL}/words/${letter}`);
  const data = await res.json();

  wordsDiv.innerHTML = "";
  data.words.forEach(word => {
    const btn = document.createElement("button");
    btn.innerText = word;
    btn.onclick = () => fetchSentence(word);
    wordsDiv.appendChild(btn);
  });

  document.getElementById("word-section").style.display = "block";
  document.getElementById("sentence-section").style.display = "none";
  document.getElementById("image-section").style.display = "none";
}

async function fetchSentence(word) {
  selectedWord = word;
  const res = await fetch(`${backendURL}/sentence/${word}`);
  const data = await res.json();
  selectedSentence = data.sentence;
  sentenceP.innerText = `"${selectedSentence}"`;

  document.getElementById("sentence-section").style.display = "block";
  document.getElementById("image-section").style.display = "none";
}

document.getElementById("generateBtn").onclick = async () => {
  const res = await fetch(`${backendURL}/generate-image`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ prompt: selectedSentence })
  });

  const blob = await res.blob();
  const imgURL = URL.createObjectURL(blob);
  document.getElementById("generatedImage").src = imgURL;
  document.getElementById("image-section").style.display = "block";
};
