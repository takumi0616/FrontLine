```bash
nohup python main.py > output.log 2>&1 &
```

タスクの削除

```bash
pkill -f "main.py"
```

権利を takumi ユーザーに指定

```bash
sudo chown -R takumi:takumi /home/takumi/docker_miniconda/src/FrontLine/
```
