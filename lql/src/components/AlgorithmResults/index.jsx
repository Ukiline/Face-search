import {useEffect, useState, useMemo, useRef} from "react";
import "./style.scss";

export default function AlgorithmResults(props) {
    const [photos, setPhotos] = useState([]);

    const showingBox = useMemo(() => {
        return (
            <div className={"showing-box"}>
                {props.first ? (
                    <h1 className={"loading"}>欢迎使用！</h1>
                ) : props.errText ? (
                    <h1 className={"loading"}>{props.errText}</h1>
                ) : (
                    photos
                )}
            </div>
        );
    }, [photos, props.errText]);

    /**
     * 渲染图片的具体过程
     * @param photoPath{[string]} 传过来的照片的文件名
     */
    const getPhotos = (photoPath) => {
        const promiseArr = [];
        for (let photoNum of photoPath) {
            promiseArr.push(
                // import(`@\\static\\imgs\\photo\\${photoNum}.jpg`) // windows兼容
                import(`../../../public/assets/imgs/photo/${photoNum}`) // mac,linux端兼容
            );
        }
        Promise.all(promiseArr).then((modules) => {
            const photos = [];
            for (let module of modules) {
                photos.push(
                    <img
                        alt="图片"
                        src={module.default}
                        key={module.default}
                        className="result"
                    />
                );
            }
            setPhotos(photos);
        });
    };

    const useDebounce = (fn, delay) => {
        const refTimer = useRef(null);
        return function f(...args) {
            if (refTimer.current) {
                clearTimeout(refTimer.current);
            }
            refTimer.current = setTimeout(() => {
                fn(args);
            }, delay);
        };
    };

    useEffect(
        useDebounce(() => {
            getPhotos(props.photoPath);
        }, 500)
        , [props.photoPath]
    );

    return (
        <div>
            {showingBox}
        </div>
    );
}
