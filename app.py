# TODO 增加一个辅助功能: 每个计算过程区附带一个链接区. 点击链接区可以快速生成一个或多个相关的计算单元格, 便于用户验算当前过程区未给出过程直接得到的结果
# TODO 后端返回 task_id， 前端引入轮询: 实现异步处理, 避免前端卡死
# TODO 注意: ab 会被解析成一个字母, 而不是 a*b. * 绝对不可以省略!!!
# TODO 把所有用于验证的 a == b 换成 a.euqals(b). 后者可以解决表达式有多个等价形式的情况
# TODO 所有括号换成 \left( 和 \right), 这样的括号更加美观, 会自动匹配表达式的高度

# pyinstaller 打包时需要添加此句
import matplotlib.backends.backend_svg

from flask import Flask
from config import TEMPLATE_FOLDER, STATIC_FOLDER
from routes import main, api


def create_app():
    app = Flask(__name__, template_folder=TEMPLATE_FOLDER,
                static_folder=STATIC_FOLDER)
    app.register_blueprint(main)
    app.register_blueprint(api)
    return app


if __name__ == '__main__':
    create_app().run(debug=True)
