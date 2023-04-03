import {useState} from "react";
import PaintingResults from "./components/PaintingResults";
import AlgorithmResults from "./components/AlgorithmResults";
import bg from "../public/assets/imgs/pexels-the-happiest-face-)-866351.jpg";
import "./App.scss";

function App() {
    const [tempPic, setTempPic] = useState([]);
    const [first, setFirst] = useState(true);
    const [errText, setErrText] = useState(null);

    /**
     * 设置是否为初始状态，通常是转换为false
     * @param status{boolean} 状态值，通常是转换为false
     */
    const handleFirst = (status) => {
        setFirst(status);
    };

    /**
     * 将图片列表从子组件传递给父组件
     * @param t{[[string]]} temp 传递的错误信息
     */
    const handlePicture = (t) => {
        setTempPic([...t]);
    };

    /**
     * 将网络错误信息传递给父组件
     * @param {string} errText
     */
    const handleErr = (errText) => {
        setErrText(errText);
    };

    return (
        <div className="App">
            <img src={bg} alt={"bg"} className={"bg-img"}/>
            <div className={"center"}>
                <h1>Sketch Less Face Image Retrieval</h1>
                <div className={"main"}>
                    <div className={"left"}>
                        <PaintingResults
                            handlePicture={handlePicture}
                            handleFirst={handleFirst}
                            handleErr={handleErr}
                        />
                    </div>
                    <div className={"right"}>
                        <AlgorithmResults
                            photoPath={tempPic}
                            first={first}
                            errText={errText}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;
