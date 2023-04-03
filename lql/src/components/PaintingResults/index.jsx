import {useEffect, useState} from "react";
import api from "../../service/api";
import emptyPic from "./emptyPic";
import "./style.scss";

const io = require("socket.io-client");
const socket = io.connect("http://10.16.45.16:4000");

export default function PaintingResults(props) {
    const [picture, setPicture] = useState(emptyPic);

    /**
     * 向后端发送请求，返回列表。
     * @param pic 要发送的图片的base64格式
     */
    const getList = (pic) => {
        if (pic === emptyPic) return;
        try {
            api.post("/sketch", pic, "image/jpeg").then(
                (res) => {
                    props.handleErr(null);
                    console.log(res.data);
                    if (res.data === undefined)
                        throw new Error("res.data is undefined");
                    let arr = res.data.join().split(",");
                    props.handlePicture(arr);
                },
                (err) => {
                    props.handleErr(`${err.status} ${err.statusText}`);
                    console.error(`${err.status} ${err.statusText}`);
                }
            );
        } catch (err) {
            props.handleErr(err);
            console.error(err);
        }
    };


    useEffect(() => {
        socket.on("painting", (res) => {
            console.log(res);
            props.handleFirst(false);
            getList(res);
            setPicture(res);
        });
    }, []);

    return (
        <div>
            <div className="picture-box">
                <img src={picture} alt={"图片未填充"} className={"inner"}/>
            </div>
        </div>
    );
}
